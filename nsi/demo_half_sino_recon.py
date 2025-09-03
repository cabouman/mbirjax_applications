import warnings
import time
import numpy as np
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp
import os


def stitch_slices(top, bottom, num_overlap_slices):
    """Stitch two reconstruction volumes along the slice (last) axis with blending.

    Args:
        top (jnp.ndarray): First ("top") half of the reconstruction.
        bottom (jnp.ndarray): Second ("bottom") half of the reconstruction.
        num_overlap_slices (int): Number of slices to overlap on each side.

    Returns:
        jnp.ndarray: Quilted reconstruction with shape
        ``[..., 2*S - 2*num_overlap_slices]`` along the last axis.
    """
    assert top.shape[:-1] == bottom.shape[:-1], "XY (non-slice) dims must match"

    # Define variables
    num_slices = top.shape[-1]               # Number of top slices
    ov = int(num_overlap_slices)    # Number of overlapping slices per side
    bl = int(ov/2)                 # Number of blended slices per side

    top_main = top[..., :num_slices - (ov + bl)]
    top_blended = top[..., num_slices - (ov + bl):num_slices - (ov - bl)]

    bot_main = bottom[..., ov + bl:]
    bot_blended = bottom[..., ov - bl: ov + bl]

    n = top_blended.shape[-1]
    w = jnp.linspace(1.0, 0.0, n).reshape((1,) * (top.ndim - 1) + (n,))
    blended = w * top_blended + (1.0 - w) * bot_blended

    return jnp.concatenate([top_main, blended, bot_main], axis=-1)


def recon_half_sino(ct_model, sino, weights=None, overlap=5):
    """Reconstruct from a full sinogram by splitting detector rows into two overlapping halves,
    reconstructing each half with its own ConeBeamModel, and quilting the halves along the
    slice axis using `stitch_slices`.

    Args:
        ct_model (mj.ConeBeamModel): A *full-geometry* ConeBeam model already configured
            (angles, distances, offsets, etc.).
        sino (jnp.ndarray | np.ndarray): Full sinogram shaped (num_views, num_rows, num_cols).
        weights (jnp.ndarray | np.ndarray, optional): Optional sinogram weights with the same
            shape as `sino`. If provided, they are split consistently and passed to recon.
        overlap (int): Number of overlapping detector rows and recon slices.
            Must satisfy 0 < overlap < num_rows, and later 0 < overlap < recon_slices for quilting.

    Returns:
        jnp.ndarray: Final quilted reconstruction volume.

    Raises:
        ValueError: If inputs are missing or shapes are inconsistent.
        AssertionError: If array dimensions are invalid.
    """
    # -------- Basic validation --------
    if ct_model is None:
        raise ValueError("ct_model must be provided.")
    if sino is None:
        raise ValueError("sino must be provided.")
    if not (hasattr(sino, "ndim") and sino.ndim == 3):
        raise AssertionError("sino must be a 3D array shaped (num_views, num_rows, num_cols).")
    if weights is not None and getattr(weights, "shape", None) != sino.shape:
        raise AssertionError("weights, if provided, must have the same shape as sino.")

    num_views, num_rows, num_cols = sino.shape

    # Validate overlap value for detector-row split
    if not isinstance(overlap, (int, np.integer)):
        raise TypeError("overlap must be an integer.")
    if not (0 < overlap < num_rows):
        raise ValueError(f"overlap must satisfy 0 < overlap < num_rows ({num_rows}).")

    # Test that model is cone beam geometry
    if not isinstance(ct_model, mj.ConeBeamModel):
        raise TypeError("ct_model must be an mbirjax ConeBeamModel.")

    # -------- parameters that will be use to create top and bottom models --------
    delta_det_row = ct_model.get_params('delta_det_row')
    det_row_offset = ct_model.get_params('det_row_offset')
    delta_voxel = ct_model.get_params('delta_voxel')

    # -------- Required parameters for cone beam geometry --------
    angles = ct_model.get_params('angles')
    source_detector_dist = ct_model.get_params('source_detector_dist')
    source_iso_dist = ct_model.get_params('source_iso_dist')

    # Optional but commonly present; guard each individually.
    optional_copy = {}
    for k in ('delta_det_channel', 'delta_det_row', 'det_row_offset', 'det_channel_offset', 'delta_voxel', 'positivity_flag', 'snr_db', 'sharpness', 'verbose'):
        try:
            optional_copy[k] = ct_model.get_params(k)
        except Exception:
            pass

    # -------- Choose an even detector row nearest isocenter --------
    det_center_row_float = ((num_rows - 1) / 2.0) + (det_row_offset / delta_det_row)
    det_center_row_index = int(np.round(det_center_row_float))
    det_center_row_index -= det_center_row_index % 2  # force even

    # -------- Row ranges for top and bottom sinogram halves --------
    top_lo = 0
    top_hi = min(det_center_row_index + overlap, num_rows)
    bot_lo = max(det_center_row_index - overlap, 0)
    bot_hi = num_rows

    # -------- Slice sinogram (and weights) halves --------
    sino_top_half = sino[:, top_lo:top_hi, :]
    sino_bot_half = sino[:, bot_lo:bot_hi, :]

    weights_top_half = None
    weights_bot_half = None
    if weights is not None:
        weights_top_half = weights[:, top_lo:top_hi, :]
        weights_bot_half = weights[:, bot_lo:bot_hi, :]

    # -------- Shapes and detector-row center alignment --------
    top_shape = (num_views, top_hi - top_lo, num_cols)
    bot_shape = (num_views, bot_hi - bot_lo, num_cols)

    det_center = (num_rows - 1) / 2.0
    top_det_center = (top_shape[1] - 1) / 2.0
    bot_det_center = (bot_shape[1] - 1) / 2.0

    top_det_row_offset = det_row_offset + ((det_center - top_lo) - top_det_center) * delta_det_row
    bot_det_row_offset = det_row_offset + ((det_center - bot_lo) - bot_det_center) * delta_det_row

    # -------- Build top-half model --------
    ct_model_top_half = mj.ConeBeamModel(
        top_shape,
        angles=angles,
        source_detector_dist=source_detector_dist,
        source_iso_dist=source_iso_dist,
    )
    if optional_copy:
        ct_model_top_half.set_params(**optional_copy)
    ct_model_top_half.set_params(det_row_offset=top_det_row_offset)

    # -------- Build bottom-half model --------
    ct_model_bot_half = mj.ConeBeamModel(
        bot_shape,
        angles=angles,
        source_detector_dist=source_detector_dist,
        source_iso_dist=source_iso_dist,
    )
    if optional_copy:
        ct_model_bot_half.set_params(**optional_copy)
    ct_model_bot_half.set_params(det_row_offset=bot_det_row_offset)

    # -------- Harmonize recon shapes --------
    top_recon_shape = ct_model_top_half.get_params('recon_shape')
    bot_recon_shape = ct_model_bot_half.get_params('recon_shape')

    # Validate overlap value against recon slice dimension for quilting
    recon_slices = int(min(top_recon_shape[2], bot_recon_shape[2]))
    if not (0 < overlap < recon_slices):
        raise ValueError(f"overlap must satisfy 0 < overlap < recon_slices ({recon_slices}).")

    # -------- Slice offsets for quilting --------
    top_recon_slice_offset = (-(top_recon_shape[2] / 2) + overlap) * delta_voxel
    bot_recon_slice_offset = ((bot_recon_shape[2] / 2) - overlap) * delta_voxel

    ct_model_top_half.set_params(recon_slice_offset=top_recon_slice_offset)
    ct_model_bot_half.set_params(recon_slice_offset=bot_recon_slice_offset)

    # -------- Reconstruct halves (pass weights if provided) --------
    recon_top_half, _ = ct_model_top_half.recon(sino_top_half, weights=weights_top_half)
    recon_bot_half, _ = ct_model_bot_half.recon(sino_bot_half, weights=weights_bot_half)

    # -------- Quilt and return --------
    recon_full = stitch_slices(recon_top_half, recon_bot_half, overlap)
    return recon_full



if __name__ == "__main__":
    print('This script demonstrates half-sinogram reconstruction.\n')

    output_path = './results'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        warnings.warn(f'Created output directory {output_path}. For faster I/O on clusters, consider symlinking to scratch, e.g.,\n'
                      f'  ln -s /scratch/gautschi/<username>/results {output_path}')

    # path to store and extract the NSI data and metadata.
    download_dir = './demo_data/'

    # NSI file path
    dataset_url = 'https://www.datadepot.rcac.purdue.edu/bouman/data/demo_nsi_vert_metal_all_views.tgz'

    # Download and extract data. Then set path to NSI scan directory.
    dataset_dir = mj.download_and_extract(dataset_url, download_dir)

    # preprocessing parameters
    downsample_factor = [8, 8]  # downsample factor of scan view images along detector rows and detector columns.
    subsample_view_factor = 8  # view subsample factor.

    # recon parameters
    sharpness = 1.0
    snr_db = 30.0

    print("\n************** NSI dataset preprocessing **************")
    sino, cone_beam_params, optional_params = mjp.nsi.compute_sino_and_params(dataset_dir,
                                                downsample_factor=downsample_factor,
                                                subsample_view_factor=subsample_view_factor)

    print("\n***************** Set up MBIRJAX model ****************")
    # Construct cone beam object using NSI parameters
    ct_model = mj.ConeBeamModel(**cone_beam_params)

    # Set optional NSI geometry parameters
    ct_model.set_params(**optional_params)

    # Set user determined parameter values
    ct_model.set_params(sharpness=sharpness, snr_db=snr_db, verbose=1)

    # Print out model parameters
    ct_model.print_params()

    print("\n***************** Reconstruct top/bottom halves ****************")
    t0 = time.time()
    recon_full = recon_half_sino(ct_model, sino)  # weights can be passed as third arg if available
    t1 = time.time()
    print(f"Stitched recon shape: {recon_full.shape}   (elapsed: {t1 - t0:.1f}s)")
    mj.slice_viewer(recon_full, slice_axis=[1, 1], title="Blended Recon")
