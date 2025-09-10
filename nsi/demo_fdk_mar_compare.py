import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import scipy
import mbirjax as mj
import mbirjax.preprocess as mjp

import pprint

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('This script is a simple demonstration of the mbirjax metal artifact reduction (MAR).\n')

    # recon parameters
    sharpness = 1.0
    snr_db = 30.0
    alpha = [1.0, 0.0, 0.0]  # BH_correction coefficient
    downsample = 4
    verbose = 1              # Print, but do not display plots

    # User defined paths
    output_path = './output/nsi_demo_mar/'   # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist
    download_dir = './demo_data/'            # Directory to store downloaded data


    # Prompt the user for dataset choice
    choice = input("Download dataset with metal? (Y/n): ").strip().lower()
    if choice == 'n':
        # URL to test phantom without metal
        dataset_url = 'https://www.datadepot.rcac.purdue.edu/bouman/data/demo_nsi_vert_no_metal_all_views.tgz'
        metal = False
    else:
        # URL to test phantom with metal
        dataset_url = 'https://www.datadepot.rcac.purdue.edu/bouman/data/demo_nsi_vert_metal_all_views.tgz'
        metal = True
    print(f"Selected dataset URL: {dataset_url}")

    # Download data
    dataset_dir = mj.download_and_extract(dataset_url, download_dir)

    # preprocessing parameters
    downsample_factor = [downsample, downsample]  # downsample factor of scan images along detector rows and detector columns.
    subsample_view_factor = 2*downsample  # view subsample factor.

    print("\n************** NSI dataset preprocessing **************")
    sino, cone_beam_params, optional_params = \
        mj.preprocess.nsi.compute_sino_and_params(dataset_dir,
                                                       downsample_factor=downsample_factor,
                                                       subsample_view_factor=subsample_view_factor)

    # beam hardening correction
    sino = jnp.maximum(sino, 0.0)
    sino = mjp.BH_correction(sino, alpha=alpha)

    print("\n************** Set up MBIRJAX model **************")
    # ConeBeamModel constructor
    ct_model = mj.ConeBeamModel(**cone_beam_params)

    # Set additional geometry arguments
    ct_model.set_params(**optional_params)

    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, verbose=1, positivity_flag=True)

    # Print out model parameters
    ct_model.print_params()

    print("\n************** Calculate transmission_root sinogram weights **************")
    weights = mj.gen_weights(sino, weight_type='transmission_root')

    print("\n************** Perform initial FDK reconstruction **************")
    recon_fdk = ct_model.fdk_recon(sino)

    print("\n************** Calculate MAR sinogram weights **************")
    weights_mar = mj.gen_weights_mar(ct_model, sino, init_recon=recon_fdk, beta=1.0, gamma=3.0)

    if verbose > 1:
        mj.slice_viewer(weights_mar, jnp.abs(sino), vmin=0, vmax=2.0, slice_axis=[0, 0], slice_label= ["Weights", "Sinogram"])

    print("\n************** Perform MBIR recon **************")
    recon_mar, recon_dict_mar = ct_model.recon(sino, weights=weights_mar)

    # Save recon to hdf5
    print("\n*********** save recon in h5 format *************")
    mar_path = os.path.join(output_path, f"recon_fdk.h5")
    mj.export_recon_hdf5(mar_path, recon_fdk, recon_dict=None)
    mar_path = os.path.join(output_path, f"recon_mar.h5")
    mj.export_recon_hdf5(mar_path, recon_mar, recon_dict=recon_dict_mar)

    if verbose > 1:
        vmin = 0
        vmax = downsample_factor[0] * 0.025
        mj.slice_viewer(recon_fdk, recon_mar, data_dicts=[None, recon_dict_mar], vmin=0, vmax=vmax, slice_label= ["FDK", "MBIR MAR"], title='Comparison')

