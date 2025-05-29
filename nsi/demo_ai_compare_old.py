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

    # output_path = './results'
    output_path = './output/nsi_demo_mar/'   # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

    # NSI file path
    dataset_url = '/depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal.tgz'
    # destination path to download and extract the NSI data and metadata.
    download_dir = './demo_data/'
    # Path to NSI scan directory.
    dataset_dir = mj.download_and_extract_tar(dataset_url, download_dir)

    # #### preprocessing parameters
    downsample_rate = [4, 4]  # downsample factor of scan images along detector rows and detector columns.
    subsample_view_factor = 8  # view subsample factor.

    # #### recon parameters
    sharpness = 1.0
    snr_db = 30.0
    alpha = [1.0, 0.0, 0.0]  # BH_correction coefficient


    print("\n*******************************************************",
          "\n************** NSI dataset preprocessing **************",
          "\n*******************************************************")
    sino, cone_beam_params, optional_params = \
        mjp.nsi.compute_sino_and_params(dataset_dir, downsample_factor=downsample_rate, subsample_view_factor=subsample_view_factor)

    # #### beam hardening correction
    sino = jnp.maximum(sino, 0.0)   # Clip sinogram to be non-negative
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
          "\n********* Perform initial FDK reconstruction **********",
          "\n*******************************************************")
    recon_fdk = ct_model.direct_recon(sino)

    print("\n*******************************************************",
          "\n*************** Estimate Metal Sinogram ***************",
          "\n*******************************************************")

    metal_sino, full_sino_estimate, metal_mask,  plastic_mask, artifact_mask = mar_utils.estimate_metal_sino_multi_material_cross(ct_model, sino, recon_fdk, verbose=1)

    # View and delete artifact and plastic mask
    mj.slice_viewer(plastic_mask, artifact_mask, slice_axis=0, slice_label=['Plastic Mask', 'Artifact Mask'], title='Plastic vs Artifact Mask Comparison')
    del artifact_mask, plastic_mask

    # Material Decomposition
    plastic_sino = sino - metal_sino
    plastic_sino = np.maximum(plastic_sino, 0.0)

    # View metal and plastic sinograms
    mj.slice_viewer(metal_sino, plastic_sino, slice_axis=0, title='The estimated metal sinogram and extracted plastic sinogram', slice_label=['Metal Sinogram', 'Plastic Sinogram'])

    print("\n*******************************************************",
          "\n************* Perform MBIR Reconstruction *************",
          "\n*******************************************************")

    weights_trans = ct_model.gen_weights(sino, weight_type='transmission_root')
    recon1,_ = ct_model.recon(plastic_sino, weights=weights_trans)

    # Fuse the metal and plastic reconstructions
    recon_mar1 = recon1 * (1.0-metal_mask) + recon_fdk * metal_mask

    print("\n*******************************************************",
          "\n*************** Estimate Metal Sinogram ***************",
          "\n*******************************************************")
    metal_sino2, full_sino_estimate2, metal_mask2, plastic_mask2, artifact_mask2 = mar_utils.estimate_metal_sino_multi_material_cross(ct_model, sino, recon_mar1, verbose=1)

    # Material Decomposition
    plastic_sino2 = sino - metal_sino2
    plastic_sino2 = np.maximum(plastic_sino2, 0.0)

    del artifact_mask2, plastic_mask2, metal_mask

    mj.slice_viewer(plastic_sino, plastic_sino2, slice_axis=0, title='Comparison between the estimated plastic sinogram from the 1st and 2nd iterations', slice_label=['Plastic Sinogram', 'Plastic Sinogram2'])

    print("\n*******************************************************",
          "\n************** Second MBIR Iteration ******************",
          "\n*******************************************************")
    recon2, _ = ct_model.recon(plastic_sino2, weights=weights_trans, max_iterations=15, init_recon=recon1)

    # Fuse the metal and plastic reconstructions
    recon_mar2 = recon2 * (1.0 - metal_mask2) + recon_fdk * metal_mask2

    recon_fdk = np.transpose(recon_fdk, axes=(2, 0, 1))
    recon_mar1 = np.transpose(recon_mar1, axes=(2, 0, 1))
    recon_mar2 = np.transpose(recon_mar2, axes=(2, 0, 1))

    vmin = 0
    vmax = downsample_rate[0] * 0.025
    mj.slice_viewer(recon_mar1, recon_mar2, vmin=0, vmax=vmax, slice_axis=0, slice_label=['MBIR MAR1', 'MBIR MAR2'], title='Comparison between the first and second MBIR')

    mj.slice_viewer(recon_fdk, recon_mar2, vmin=0, vmax=vmax, slice_axis=0, slice_label=['FDK', 'MBIR MAR2'], title='Comparison between FDK and the second MBIR')

    # Compare the plastic region
    thresholds_fdk = mj.multi_threshold_otsu(recon_fdk, classes=3)
    thresholds_mbir = mj.multi_threshold_otsu(recon_mar2, classes=3)

    # Assign thresholds
    plastic_thresh_fdk = thresholds_fdk[0]
    metal_thresh_fdk = thresholds_fdk[1]

    plastic_thresh_mbir = thresholds_mbir[0]
    metal_thresh_mbir = thresholds_mbir[1]

    # Segment plastic masks
    plastic_mask_fdk = jnp.where((recon_fdk > plastic_thresh_fdk) & (recon_fdk <= metal_thresh_fdk), 1.0, 0.0)
    plastic_mask_mbir = jnp.where((recon_mar2 > plastic_thresh_mbir) & (recon_mar2 <= metal_thresh_mbir), 1.0, 0.0)

    # Visualize
    mj.slice_viewer(plastic_mask_fdk, plastic_mask_mbir, slice_axis=0, slice_label=['FDK Plastic', 'MBIR Plastic'], title='Plastic Region Comparison (FDK vs MBIR)')
