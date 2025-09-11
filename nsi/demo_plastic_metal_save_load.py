import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import scipy
import argparse
import mbirjax as mj
import mbirjax.preprocess as mjp
import pickle

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('This script demonstrates mbirjax metal-plastic reconstruction.\n')

    load_sino = False
    save_sino = True

    # === MBIR Recon parameters ===
    sharpness = 1.0
    num_metal = 2  # Number of distinct metal materials
    alpha = [1.0, 0.0, 0.0]  # beam_hardening_correction coefficient
    downsample = 4  # Default down sampling rate
    verbose = 1  # Print, but do not display

    # ----------------------------
    # Parse command line arguments
    # ----------------------------
    parser = argparse.ArgumentParser(description="MBIRJAX Plastic-Metal Reconstruction Demo")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to existing data directory.")
    parser.add_argument("--downsampling", type=int, default=None,
                        help="Downsampling factor (sets detector and view downsampling).")
    args = parser.parse_args()

    # Output path
    output_path = './output/lilly/'  # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

    # === Choose dataset ===
    dataset_choice = "AI"

    # Options:
    #   "AI"             -> Autoinjector HighRes Horizontal
    #   "CAI_horizontal" -> Connected Autoinjector Horizontal
    #   "CAI_vertical"   -> Connected Autoinjector Vertical
    #   "existing_data"  -> Use uncompressed scan data in the folder pointed to by existing_directory

    if dataset_choice == "AI":
        dataset_url = '/depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal.tgz'
        dataset_tag = 'ai'
    elif dataset_choice == "CAI_horizontal":
        dataset_url = '/depot/bouman/data/Lilly/Connected_Autoinjector_Horizontal.tgz'
        dataset_tag = 'cai_h'
    elif dataset_choice == "CAI_vertical":
        dataset_url = '/depot/bouman/data/Lilly/Connected_Autoinjector_Vertical.tgz'
        dataset_tag = 'cai_v'
    else:
        raise ValueError(f"Unknown dataset choice: {dataset_choice}")

    # Destination path to download and extract the NSI data and metadata.
    download_dir = './demo_data/'

    if args.data_path is not None and not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"--data_path does not exist or is not a directory: {args.data_path}")
    if args.data_path is not None:
        dataset_dir = args.data_path
        dataset_tag = os.path.basename(dataset_dir.rstrip("/"))
    else:
        dataset_dir = mj.download_and_extract(dataset_url, download_dir)

    # Override default down sampling rate if provided
    if args.downsampling is not None:
        downsample = args.downsampling

    # Set down sampling rates
    downsample_rate = [downsample, downsample]
    subsample_view_factor = 2 * downsample

    sino_path = os.path.join(output_path, f"sino_{dataset_tag}.npz")
    cb_params_path = os.path.join(output_path, f"cb_params_{dataset_tag}.pkl")
    opt_params_path = os.path.join(output_path, f"opt_params_{dataset_tag}.pkl")

    if load_sino:
        sino = np.load(sino_path)['sino']
        with open(cb_params_path, 'rb') as f:
            cone_beam_params = pickle.load(f)
        with open(opt_params_path, 'rb') as f:
            optional_params = pickle.load(f)
        print('Loaded sino from {}'.format(sino_path))
        print('Sinogram shape = {}'.format(sino.shape))
    else:
        print("\n************** NSI dataset preprocessing **************")
        sino, cone_beam_params, optional_params = \
            mjp.nsi.compute_sino_and_params(dataset_dir, downsample_factor=downsample_rate,
                                            subsample_view_factor=subsample_view_factor)

        # #### beam hardening correction
        sino = mjp.BH_correction(sino, alpha=alpha)
        sino = jnp.maximum(sino, 0.0)  # Clip sinogram to be non-negative

    if save_sino:
        np.savez_compressed(sino_path, sino=sino)
        with open(cb_params_path, 'wb') as f:
            pickle.dump(cone_beam_params, f)
        with open(opt_params_path, 'wb') as f:
            pickle.dump(optional_params, f)

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

    # Visualize masks
    if verbose >= 2:
        labels = ['Plastic Mask'] + [f'Metal {i + 1} Mask' for i in range(len(metal_masks))]
        mj.slice_viewer(plastic_mask, *metal_masks, vmin=0, vmax=1.0, slice_axis=0,
                        slice_label=labels, title="Final Plastic and Metal Masks")

    # Compute FDK reconstruction
    recon_fdk = ct_model.direct_recon(sino)

    # Save recon to hdf5
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
