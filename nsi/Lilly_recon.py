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
    # === MBIR Recon parameters ===
    sharpness = 1.0
    verbose = 1

    # ----------------------------
    # Parse command line arguments
    # ----------------------------
    parser = argparse.ArgumentParser(description="MBIRJAX Plastic-Metal Reconstruction Demo")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to existing data directory.")
    parser.add_argument("--downsampling", type=int, default=1,  # Perhaps change to subsample_detector_factor
                        help="Subsampling factor for detector rows and channels.")
    parser.add_argument("--subsample_view_factor", type=int, default=1,
                        help="Subsampling factor for projection views.")
    parser.add_argument("--num_metal", type=int, default=2,
                        help="Number of metal types for segmentation and MAR.")
    parser.add_argument("sino_cropping", type=int, default=1,
                        help="Flag for applying sinogram cropping")
    args = parser.parse_args()

    # Set output path
    output_path = './output/lilly/'   # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

    if args.data_path is not None and not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"--data_path does not exist or is not a directory: {args.data_path}")

    # Get parameters from command line
    dataset_dir = args.data_path
    downsample = args.downsampling
    num_metal = args.num_metal
    subsample_view_factor = args.subsample_view_factor

    # Set program parameters
    downsample_rate = [downsample, downsample]
    dataset_tag = os.path.basename(dataset_dir.rstrip("/"))

    if verbose>0:
        print("\n************** NSI dataset preprocessing **************")
    sino, cone_beam_params, optional_params = \
        mjp.nsi.compute_sino_and_params(dataset_dir, downsample_factor=downsample_rate, subsample_view_factor=subsample_view_factor)
    cropping = bool(args.sino_cropping)
    if cropping:
        sino, cone_beam_params, optional_params = mjp.auto_crop_sino_conebeam(sino, cone_beam_params, optional_params)
        if verbose>0:
            print("Cropping unused sinogram margins and update cone-beam geometry parameters.")
    # Clip sinogram to be positive
    sino = jnp.maximum(sino, 0.0)   # Clip sinogram to be non-negative

    if verbose>0:
        print("\n***************** Set up MBIRJAX model ****************")
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    ct_model.set_params(**optional_params)
    ct_model.set_params(sharpness=sharpness, verbose=verbose, positivity_flag=True)
    weights_trans = mj.gen_weights(sino, weight_type='transmission_root')
    if verbose>0:
        ct_model.print_params()

    if verbose>0:
        print("\n*************** Compute reconstruction ***************")
    recon = mjp.recon_plastic_metal(ct_model, sino, weights_trans, num_metal=num_metal, verbose=verbose)

    # Save recon to hdf5
    if verbose>0:
        print("\n*********** save mar and fdk recon in h5 format *************")
    mar_path = os.path.join(output_path, f"recon_{dataset_tag}_nummetal_{num_metal}_mar.h5")
    mj.export_recon_hdf5(mar_path, recon, recon_dict=None, remove_flash=True)
    if verbose>0:
        print("Metal artifact reduction recon saved to {}".format(os.path.abspath(mar_path)))

