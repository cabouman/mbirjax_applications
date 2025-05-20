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
    print('This script is a demonstration of the mbirjax metal artifact reduction (MAR) capability.\
    \n Demo functionality includes:\
    \n\t * downloading NSI dataset from specified urls;\
    \n\t * Computing sinogram data;\
    \n\t * Computing the FDK reconstruction;\
    \n\t * Computing the estimated plastic and metal sinograms;\
    \n\t * Computing the inital MBIR plastic reconstruction;\
    \n\t * Computing the MAR weights;\
    \n\t * Computing the generalized Huber weights;\
    \n\t * Computing the final MBIR plastic reconstruction;\
    \n\t * Blending the plastic and metal reconstructions together;\
    \n\t * Displaying the results.\n')
    # ###################### User defined params. Change the parameters below for your own use case.
    output_path = './output/nsi_demo_mar/'  # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

    # ##### params for dataset downloading. User may change these parameters for their own datasets.
    # An example NSI dataset (tarball) will be downloaded from `dataset_url`, and saved to `download_dir`.
    # url to NSI dataset.
    #dataset_url = 'https://www.datadepot.rcac.purdue.edu/bouman/data/mar_demo_data.tgz'
    dataset_url = '/depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal.tgz'
    # destination path to download and extract the NSI data and metadata.
    download_dir = './demo_data/'
    # Path to NSI scan directory.
    dataset_dir = mj.download_and_extract_tar(dataset_url, download_dir)
    # for testing user prompt in NSI preprocessing function
    # dataset_dir = "/depot/bouman/data/share_conebeam_data/Autoinjection-Full-LowRes/Vertical-0.5mmTin"

    # #### preprocessing parameters
    downsample_factor = [4, 4]  # downsample factor of scan images along detector rows and detector columns.
    subsample_view_factor = 1  # view subsample factor.

    # #### recon parameters
    sharpness = 1.0
    snr_db = 30.0
    alpha = [1.0, 0.0, 0.0]  # beam_hardening_correction coefficient


    print("\n*******************************************************",
          "\n************** NSI dataset preprocessing **************",
          "\n*******************************************************")
    sino, cone_beam_params, optional_params = \
        mjp.nsi.compute_sino_and_params(dataset_dir,
                                                       downsample_factor=downsample_factor,
                                                       subsample_view_factor=subsample_view_factor)

    # #### beam hardening correction
    sino = jnp.maximum(sino, 0.0)
    sino = mar_utils.beam_hardening_correction(sino, alpha=alpha)

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
          "\n*************** Estimate Metal Sinogram ***************",
          "\n*******************************************************")
    metal_sino, metal_mask = mar_utils.estimate_metal_sino(ct_model, sino, fdk_recon, verbose=1)
    plastic_sino = sino - metal_sino
    mj.slice_viewer(plastic_sino, metal_sino, vmin=0, vmax=2.0, slice_axis=[0, 0], slice_label= ["Plastic Sino", "Metal Sino"])

    print("\n*******************************************************",
          "\n************ Calculate MAR sinogram weights ***********",
          "\n*******************************************************")
    weights_mar = ct_model.gen_weights_mar(sino, init_recon=fdk_recon, beta=1.0, gamma=3.0)
    mj.slice_viewer(weights_mar, jnp.abs(sino), vmin=0, vmax=2.0, slice_axis=[0, 0], slice_label=["Weights", "Sinogram"])

    print("\n*******************************************************",
          "\n******** Perform MBIR recon with MAR weights **********",
          "\n*******************************************************")
    recon_plastic, recon_params = ct_model.recon(plastic_sino, weights=weights_mar)

    print("\n*******************************************************",
          "\n*********** Blend metal and plastic recons ************",
          "\n*******************************************************")
    recon_mar = recon_plastic * (1.0-metal_mask) + fdk_recon * metal_mask

    # #### Display results
    # change the image data shape to (slices, rows, cols)
    fdk_recon = np.transpose(fdk_recon, axes=(2, 0, 1))
    recon_mar = np.transpose(recon_mar, axes=(2, 0, 1))

    vmin = 0
    vmax = downsample_factor[0] * 0.025
    mj.slice_viewer(fdk_recon, recon_mar, vmin=0, vmax=vmax, slice_axis=[0, 0], slice_label=["FDK", "MBIR MAR"])
