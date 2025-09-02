import warnings
import time
import numpy as np
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp
import os


def quilt_slices(top, bottom, num_overlap_slices):
    """Quilt two reconstruction volumes along the slice (last) axis.

    This function discards the last ``num_overlap_slices`` slices from
    ``top`` and the first ``num_overlap_slices`` slices from ``bottom``,
    then concatenates the results to form a continuous volume.

    Args:
        top (jnp.ndarray): First ("top") half of the reconstruction.
            Must have shape [..., S].
        bottom (jnp.ndarray): Second ("bottom") half of the reconstruction.
            Must have the same shape as ``top``.
        num_overlap_slices (int): Number of overlapping slices to discard
            from each half before concatenation.

    Returns:
        jnp.ndarray: Quilted reconstruction with shape
        ``[..., 2*S - 2*num_overlap_slices]`` along the last axis.
    """
    assert top.shape[:-1] == bottom.shape[:-1], "XY (non-slice) dims must match"
    S = top.shape[-1]
    ov = int(num_overlap_slices)
    assert 0 < ov <= S, f"num_overlap_slices must be in [1, {S}]"

    # Non-overlap parts
    top_main = top[..., :S - ov]
    bot_main = bottom[..., ov:]

    return jnp.concatenate([top_main, bot_main], axis=-1)



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

    # Get needed recon parameters
    num_views, num_rows, num_cols = sino.shape
    recon_shape = ct_model.get_params('recon_shape')
    delta_det_channel = ct_model.get_params('delta_det_channel')
    delta_det_row = ct_model.get_params('delta_det_row')
    det_channel_offset = ct_model.get_params('det_channel_offset')
    det_row_offset = ct_model.get_params('det_row_offset')
    delta_voxel = ct_model.get_params('delta_voxel')
    source_detector_dist = ct_model.get_params('source_detector_dist')
    source_iso_dist = ct_model.get_params('source_iso_dist')
    angles = ct_model.get_params('angles')
    magnification = ct_model.get_magnification()

    # choose an even detector row nearest isocenter
    det_center_row_float = ((num_rows - 1) / 2.0) + (det_row_offset / delta_det_row)
    det_center_row_index = int(np.round(det_center_row_float))
    det_center_row_index -= det_center_row_index % 2  # force even

    # Set amount of overlap in detector and recon space
    det_overlap_rows = 5  # overlap on each side
    recon_overlap_slices = 5  # overlap on each side

    # Calculate row ranges for top and bottom sinogram halves
    top_lo = 0
    top_hi = min(det_center_row_index + det_overlap_rows, num_rows)

    bot_lo = max(det_center_row_index - det_overlap_rows, 0)
    bot_hi = num_rows

    # Construct sinogram halves
    sino_top_half = sino[:, top_lo:top_hi, :]
    sino_bot_half = sino[:, bot_lo:bot_hi, :]

    # Compute shape of sinogram halves
    top_shape = (num_views, top_hi - top_lo, num_cols)
    bot_shape = (num_views, bot_hi - bot_lo, num_cols)

    # Compute the centers of each sinogram
    det_center = (num_rows-1)/2
    top_det_center = (top_shape[1] - 1)/2
    bot_det_center = (bot_shape[1] - 1)/2

    # Compute row offsets for each sinogram half
    top_det_row_offset = det_row_offset + ((det_center - top_lo) - top_det_center) * delta_det_row
    bot_det_row_offset = det_row_offset + ((det_center - bot_lo) - bot_det_center) * delta_det_row

    # Construct model for upper sinogram half
    ct_model_top_half = mj.ConeBeamModel(
        top_shape,
        angles=angles,
        source_detector_dist=source_detector_dist,
        source_iso_dist=source_iso_dist
    )
    ct_model_top_half.set_params(**optional_params)
    ct_model_top_half.set_params(det_row_offset=top_det_row_offset)
    ct_model_top_half.set_params(sharpness=sharpness, snr_db=snr_db, verbose=1)
    top_recon_shape = ct_model_top_half.get_params('recon_shape')
    print(f"Top-half recon shape: {top_recon_shape}")

    # Construct model for lower sinogram half
    ct_model_bot_half = mj.ConeBeamModel(
        bot_shape,
        angles=angles,
        source_detector_dist=source_detector_dist,
        source_iso_dist=source_iso_dist,
    )
    ct_model_bot_half.set_params(**optional_params)
    ct_model_bot_half.set_params(det_row_offset=bot_det_row_offset)
    ct_model_bot_half.set_params(sharpness=sharpness, snr_db=snr_db, verbose=1)
    bot_recon_shape = ct_model_bot_half.get_params('recon_shape')
    print(f"Bottom-half recon shape: {bot_recon_shape}")

    # ToDo: Figure out why the program crashs when 122 is changed to 124
    half_recon_shape = (187, 187, 122)
    print(f"Max recon shape: {half_recon_shape}")

    # Set recon shape of top and bottom half the same
    ct_model_top_half.set_params(recon_shape=half_recon_shape)
    ct_model_bot_half.set_params(recon_shape=half_recon_shape)

    # Compute slice offsets for each sinogram half
    top_recon_slice_offset = (-(half_recon_shape[2]/2) + recon_overlap_slices) * delta_voxel
    bot_recon_slice_offset = ((half_recon_shape[2]/2) - recon_overlap_slices) * delta_voxel

    # Set recon slice offsets for top and bottom half
    ct_model_top_half.set_params(recon_slice_offset=top_recon_slice_offset)
    ct_model_bot_half.set_params(recon_slice_offset=bot_recon_slice_offset)

    print("\n***************** Reconstruct top/bottom halves ****************")
    t0 = time.time()
    recon_top_half, recon_top_dict = ct_model_top_half.recon(sino_top_half)
    t1 = time.time()
    recon_bot_half, recon_bot_dict = ct_model_bot_half.recon(sino_bot_half)
    t2 = time.time()

    print(f"Top-half recon shape: {recon_top_half.shape}   (elapsed: {t1 - t0:.1f}s)")
    print(f"Bottom-half recon shape: {recon_bot_half.shape} (elapsed: {t2 - t1:.1f}s)")

    # Blend the two halves along slice axis
    recon_full = quilt_slices(recon_top_half, recon_bot_half, recon_overlap_slices)
    print("Quilted (blended) recon shape:", recon_full.shape)
    mj.slice_viewer(recon_full, title="Blended Recon")

    # Put the two half recons along a new axis and show side-by-side
    title = "Top half recon (left) vs Bottom half recon (right)"

    mj.slice_viewer(recon_top_half, recon_bot_half, data_dicts=[recon_top_dict, recon_bot_dict], title=title)
