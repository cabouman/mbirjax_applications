import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import scipy
import mbirjax as mj
import mbirjax.preprocess as mjp

import mar_utils
import pprint

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('This script is a simple demonstration of the mbirjax metal artifact reduction (MAR).\n')

    # User defined params.
    output_path = './output/nsi_demo_mar/'  # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

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

    # destination path to download and extract the NSI data and metadata.
    download_dir = './demo_data/'
    # Path to NSI scan directory.
    dataset_dir = mj.download_and_extract_tar(dataset_url, download_dir)

    # preprocessing parameters
    downsample_factor = [4, 4]  # downsample factor of scan images along detector rows and detector columns.
    subsample_view_factor = 8  # view subsample factor.

    # recon parameters
    sharpness = 1.0
    snr_db = 30.0
    alpha = [1.0, 0.0, 0.0]  # BH_correction coefficient


    print("\n*******************************************************",
          "\n************** NSI dataset preprocessing **************",
          "\n*******************************************************")
    sino, cone_beam_params, optional_params = \
        mj.preprocess.nsi.compute_sino_and_params(dataset_dir,
                                                       downsample_factor=downsample_factor,
                                                       subsample_view_factor=subsample_view_factor)

    # beam hardening correction
    sino = jnp.maximum(sino, 0.0)
    sino = mjp.BH_correction(sino, alpha=alpha)

    print("\n*******************************************************",
          "\n***************** Set up MBIRJAX model ****************",
          "\n*******************************************************")
    # ConeBeamModel constructor
    ct_model = mj.ConeBeamModel(**cone_beam_params)

    # Set additional geometry arguments
    ct_model.set_params(**optional_params)

    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, verbose=1, positivity_flag=True)

    # Print out model parameters
    ct_model.print_params()

    print("\n*******************************************************",
          "\n***** Calculate transmission_root sinogram weights ****",
          "\n*******************************************************")
    weights = ct_model.gen_weights(sino, weight_type='transmission_root')

    print("\n*******************************************************",
          "\n********* Perform initial FDK reconstruction **********",
          "\n*******************************************************")
    fdk_recon = ct_model.fdk_recon(sino)

    print("\n*******************************************************",
          "\n************ Calculate MAR sinogram weights ***********",
          "\n*******************************************************")
    weights_mar = ct_model.gen_weights_mar(sino, init_recon=fdk_recon, beta=1.0, gamma=3.0)
    mj.slice_viewer(weights_mar, jnp.abs(sino), vmin=0, vmax=2.0, slice_axis=[0, 0], slice_label= ["Weights", "Sinogram"])

    print("\n*******************************************************",
          "\n******** Perform MBIR recon with MAR weights **********",
          "\n*******************************************************")
    recon_mar, recon_dict_mar = ct_model.recon(sino, weights=weights_mar)


    # #### Display results
    # change the image data shape to (slices, rows, cols)
    fdk_recon = np.transpose(fdk_recon, axes=(2, 0, 1))
    recon_mar = np.transpose(recon_mar, axes=(2, 0, 1))

    vmin = 0
    vmax = downsample_factor[0] * 0.025
    mj.slice_viewer(fdk_recon, recon_mar, attribute_dicts=[None, recon_dict_mar], vmin=0, vmax=vmax, slice_label= ["FDK", "MBIR MAR"], title='Comparison')

