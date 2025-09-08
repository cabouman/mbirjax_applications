import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import scipy
import mbirjax as mj
import mbirjax.preprocess as mjp

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('This script demonstrates mbirjax metal-plastic reconstruction.\n')

    # Change this to point to your own data and set dataset_choice="existing_data" below
    #existing_directory = "/depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal"
    existing_directory = "/home/c268681/mbirjax_applications/CAI_Horizontal"

    # Output path
    output_path = './output/lilly/'   # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

    # === Choose dataset ===
    dataset_choice = "existing_data"
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
    elif dataset_choice == "existing_data":
        dataset_url = None
        dataset_tag = os.path.basename(existing_directory)
    else:
        raise ValueError(f"Unknown dataset choice: {dataset_choice}")

    # Destination path to download and extract the NSI data and metadata.
    download_dir = './demo_data/'

    # Download data to directory.
    if dataset_choice == "existing_data":
        dataset_dir = existing_directory
    else:
        dataset_dir = mj.download_and_extract_tar(dataset_url, download_dir)

    # === Preprocessing parameters ===
    downsample_rate = [4, 4]  # downsample factor of scan images along detector rows and detector columns.
    subsample_view_factor = 4 # view subsample factor.

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
    # ConeBeamModel constructor
    ct_model = mj.ConeBeamModel(**cone_beam_params)

    # Set additional geometry arguments
    ct_model.set_params(**optional_params)
    ct_model.set_params(sharpness=sharpness, verbose=verbose, positivity_flag=True)
    weights_trans = mj.gen_weights(sino, weight_type='transmission_root')

    # Print out model parameters
    ct_model.print_params()

    print("\n*************** Compute MAR reconstruction ***************")
    recon = mjp.recon_BH_plastic_metal(ct_model, sino, weights_trans, num_metal=num_metal, verbose=verbose)

    print("\n*********** Segment plastic and metal masks *************")
    plastic_mask, metal_masks, plastic_scale, metal_scales = \
        mjp.segment_plastic_metal(recon, num_metal=num_metal)

    # Visualize masks
    if verbose >= 2:
        labels = ['Plastic Mask'] + [f'Metal {i+1} Mask' for i in range(len(metal_masks))]
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
        print("\n*********** view original and corrected reconstruction *************")
        vmin = 0
        vmax = downsample_rate[0] * 0.025
        mj.slice_viewer(recon_fdk, recon, vmin=0, vmax=vmax, slice_axis=0, slice_label=['FDK', 'MBIR MAR'], title='Comparison between the original and corrected reconstruction')
