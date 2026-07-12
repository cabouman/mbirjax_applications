import os
import pprint
import argparse
import mbirjax as mj
import mbirjax.preprocess as mjp
import jax.numpy as jnp

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('This script demonstrates mbirjax metal-plastic reconstruction.\n')

    # === MBIR Recon parameters ===
    sharpness = 1.0
    alpha = [1.0, 0.0, 0.0]           # beam_hardening_correction coefficient
    verbose = 1                       # Print, but do not display

    # ----------------------------
    # Parse command line arguments
    # ----------------------------
    parser = argparse.ArgumentParser(description="MBIRJAX Plastic-Metal Reconstruction Demo")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to existing data directory.")
    parser.add_argument("--downsampling", type=int, default=4,  # Perhaps change to subsample_detector_factor
                        help="Subsampling factor for detector rows and channels.")
    parser.add_argument("--subsample_view_factor", type=int, default=4,
                        help="Subsampling factor for projection views.")
    parser.add_argument("--num_metal", type=int, default=None,
                        help="Number of metal types for segmentation and MAR. "
                             "Defaults to the recommended value for the selected dataset.")
    parser.add_argument("--dataset", type=str, default="public",
                        choices=["public", "AI", "CAI_horizontal", "CAI_vertical"],
                        help="Dataset to use. 'public' is a publicly available demo dataset; "
                             "the others are internal Lilly test datasets.")
    args = parser.parse_args()

    # Output path
    output_path = './output/lilly/'   # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

    # === Choose dataset ===
    dataset_choice = args.dataset

    # Options:
    #   "public"         -> Publicly available demo dataset (default)
    #   "AI"             -> Autoinjector HighRes Horizontal (internal)
    #   "CAI_horizontal" -> Connected Autoinjector Horizontal (internal)
    #   "CAI_vertical"   -> Connected Autoinjector Vertical (internal)

    if dataset_choice == "public":
        dataset_url = 'https://www.datadepot.rcac.purdue.edu/bouman/data/demo_nsi_vert_metal_all_views.tgz'
        dataset_tag = 'public'
        default_num_metal = 1
    elif dataset_choice == "AI":
        dataset_url = '/depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal.tgz'
        dataset_tag = 'ai'
        default_num_metal = 2
    elif dataset_choice == "CAI_horizontal":
        dataset_url = '/depot/bouman/data/Lilly/Connected_Autoinjector_Horizontal.tgz'
        dataset_tag = 'cai_h'
        default_num_metal = 2
    elif dataset_choice == "CAI_vertical":
        dataset_url = '/depot/bouman/data/Lilly/Connected_Autoinjector_Vertical.tgz'
        dataset_tag = 'cai_v'
        default_num_metal = 2
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
    downsample = args.downsampling

    num_metal = args.num_metal if args.num_metal is not None else default_num_metal

    # Set down sampling rates
    downsample_rate = [downsample, downsample]
    subsample_view_factor = args.subsample_view_factor

    print("\n************** NSI dataset preprocessing **************")
    sino, cone_beam_params, optional_params = \
        mjp.nsi.compute_sino_and_params(dataset_dir, downsample_factor=downsample_rate, subsample_view_factor=subsample_view_factor)

    # #### beam hardening correction
    sino = mjp.BH_correction(sino, alpha=alpha)
    sino = jnp.maximum(sino, 0.0)   # Clip sinogram to be non-negative

    print("\n***************** Set up MBIRJAX model ****************")
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    ct_model.set_params(**optional_params)
    # Recompute the automatic recon geometry now that the real detector pitches and
    # offsets are set (see mbirjax.preprocess.nsi.compute_sino_and_params).
    ct_model.auto_set_recon_geometry()
    ct_model.set_params(sharpness=sharpness, verbose=verbose, positivity_flag=True)
    weights_trans = mj.gen_weights(sino, weight_type='transmission_root')
    ct_model.print_params()

    print("\n*************** Compute reconstruction ***************")
    recon = mjp.recon_plastic_metal(ct_model, sino, weights_trans, num_metal=num_metal, verbose=verbose)

    # Compute FDK reconstruction
    recon_fdk = ct_model.direct_recon(sino)

    # Save recon to hdf5
    print("\n*********** save mar and fdk recon in h5 format *************")
    mar_path = os.path.join(output_path, f"recon_{dataset_tag}_nummetal_{num_metal}_mar.h5")
    mj.export_recon_hdf5(mar_path, recon, recon_dict=None, remove_flash=True)
    fdk_path = os.path.join(output_path, f"recon_{dataset_tag}_fdk.h5")
    mj.export_recon_hdf5(fdk_path, recon_fdk, recon_dict=None)
    print("Metal artifact reduction recon saved to {}".format(os.path.abspath(mar_path)))
    print("FDK recon saved to {}".format(os.path.abspath(fdk_path)))

    if verbose >= 2:
        vmax = downsample_rate[0] * 0.025
        mj.slice_viewer(recon_fdk, recon, vmin=0, vmax=vmax, slice_axis=0,
                        slice_label=['FDK', 'MBIR MAR'],
                        title='Comparison between the original and corrected reconstruction')
