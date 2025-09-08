import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import scipy
import argparse
import mbirjax as mj
import mbirjax.preprocess as mjp

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('This script demonstrates mbirjax metal-plastic reconstruction.\n')

    # ----------------------------
    # Parse command line arguments
    # ----------------------------
    parser = argparse.ArgumentParser(description="MBIRJAX Plastic-Metal Reconstruction Demo")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to existing data directory (overrides dataset_choice=existing_data).")
    parser.add_argument("--downsampling", type=int, default=None,
                        help="Downsampling factor (sets both detector downsampling and view subsampling).")
    args = parser.parse_args()

    # Default configuration
    existing_directory = "./demo_data/CAI_Horizontal"
    output_path = './output/lilly/'
    os.makedirs(output_path, exist_ok=True)

    dataset_choice = "existing_data"
    dataset_url = None
    dataset_tag = os.path.basename(existing_directory)

    # Override with command-line arguments if provided
    if args.data_path is not None:
        dataset_choice = "existing_data"
        existing_directory = args.data_path
        dataset_tag = os.path.basename(existing_directory.rstrip("/"))

    if dataset_choice == "existing_data":
        dataset_dir = existing_directory
    else:
        raise ValueError(f"Unsupported dataset choice {dataset_choice} when using arguments.")

    download_dir = './demo_data/'

    # === Preprocessing parameters ===
    if args.downsampling is not None:
        downsample_rate = [args.downsampling, args.downsampling]
        subsample_view_factor = 2 * args.downsampling
    else:
        downsample_rate = [4, 4]   # default
        subsample_view_factor = 4  # default

    # === MBIR Recon parameters ===
    sharpness = 1.0
    num_metal = 2                     # Number of distinct metal materials
    alpha = [1.0, 0.0, 0.0]           # beam_hardening_correction coefficient

    verbose = 1

    print("\n************** NSI dataset preprocessing **************")
    sino, cone_beam_params, optional_params = \
        mjp.nsi.compute_sino_and_params(dataset_dir, downsample_factor=downsample_rate, subsample_view_factor=subsample_view_factor)

    # #### beam hardening correction
    sino = mjp.BH_correction(sino, alpha=alpha)
    sino = jnp.maximum(sino, 0.0)   # Clip sinogram to be non-negative

    print("\n***************** Set up MBIRJAX model ****************")
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    ct_model.set_params(**optional_params)
    ct_model.set_params(sharpness=sharpness, verbose=verbose, positivity_flag=True)
    weights_trans = mj.gen_weights(sino, weight_type='transmission_root')
    ct_model.print_params()

    print("\n*************** Compute MAR reconstruction ***************")
    recon = mjp.recon_BH_plastic_metal(ct_model, sino, weights_trans, num_metal=num_metal, verbose=verbose)

    print("\n*********** Segment plastic and metal masks *************")
    plastic_mask, metal_masks, plastic_scale, metal_scales = \
        mjp.segment_plastic_metal(recon, num_metal=num_metal)

    if verbose >= 2:
        labels = ['Plastic Mask'] + [f'Metal {i+1} Mask' for i in range(len(metal_masks))]
        mj.slice_viewer(plastic_mask, *metal_masks, vmin=0, vmax=1.0, slice_axis=0,
                        slice_label=labels, title="Final Plastic and Metal Masks")

    # Compute FDK reconstruction
    recon_fdk = ct_model.direct_recon(sino)

    print("\n*********** save mar and fdk recon in h5 format *************")
    mar_path = os.path.join(output_path, f"recon_{dataset_tag}_mar.h5")
    mj.export_recon_hdf5(mar_path, recon, recon_dict=None)
    fdk_path = os.path.join(output_path, f"recon_{dataset_tag}_fdk.h5")
    mj.export_recon_hdf5(fdk_path, recon_fdk, recon_dict=None)
    print("Metal artifact reduction recon saved to {}".format(os.path.abspath(mar_path)))
    print("FDK recon saved to {}".format(os.path.abspath(fdk_path)))

    if verbose >= 2:
        vmax = downsample_rate[0] * 0.025
        mj.slice_viewer(recon_fdk, recon, vmin=0, vmax=vmax, slice_axis=0,
                        slice_label=['FDK', 'MBIR MAR'],
                        title='Comparison between the original and corrected reconstruction')
