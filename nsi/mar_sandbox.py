import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import scipy
import mbirjax
import demo_utils
import mar_utils
import pprint

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('This script is a demonstration of the metal artifact reduction (MAR) functionality using MAR sinogram weight.\
    \n Demo functionality includes:\
    \n\t * downloading NSI dataset from specified urls;\
    \n\t * Computing sinogram data;\
    \n\t * Computing two sets of sinogram weights, one with type "transmission_root" and the other with type "MAR";\
    \n\t * Computing two sets of MBIR reconstructions with each sinogram weight respectively;\
    \n\t * Displaying the results.\n')
    # ###################### User defined params. Change the parameters below for your own use case.
    output_path = './output/nsi_demo_mar/'  # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

    # ##### params for dataset downloading. User may change these parameters for their own datasets.
    # An example NSI dataset (tarball) will be downloaded from `dataset_url`, and saved to `download_dir`.
    # url to NSI dataset.
    dataset_url = 'https://www.datadepot.rcac.purdue.edu/bouman/data/mar_demo_data.tgz'
    # destination path to download and extract the NSI data and metadata.
    download_dir = './demo_data/'
    # Path to NSI scan directory.
    dataset_dir = demo_utils.download_and_extract_tar(dataset_url, download_dir)
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
        mbirjax.preprocess.nsi.compute_sino_and_params(dataset_dir,
                                                       downsample_factor=downsample_factor,
                                                       subsample_view_factor=subsample_view_factor)

    # #### beam hardening correction
    sino = jnp.maximum(sino, 0.0)
    sino = mar_utils.beam_hardening_correction(sino, alpha=alpha)

    print("\n*******************************************************",
          "\n***************** Set up MBIRJAX model ****************",
          "\n*******************************************************")
    # ConeBeamModel constructor
    ct_model = mbirjax.ConeBeamModel(**cone_beam_params)

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
          "\n**** Perform recon with transmission_root weights. ****",
          "\n*******************************************************")
    print("This recon will be used to identify metal voxels and compute the MAR sinogram weight.")
    # ##########################
    # Perform VCD reconstruction
    time0 = time.time()
    # Using FDK reconstruction as initialization of VCD
    print('Starting fdk')
    fdk_recon = ct_model.fdk_recon(sino)
    elapsed = time.time() - time0
    print('Elapsed time for fdk is {:.3f} seconds'.format(elapsed))
    time0 = time.time()
    init_recon, recon_params = ct_model.recon(sino, weights=weights, init_recon=fdk_recon)
    init_recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for initial trans weight VCD recon is {:.3f} seconds'.format(elapsed))
    # ##########################

    print("\n*******************************************************",
          "\n*************** Estimate Metal Sinogram ***************",
          "\n*******************************************************")
    metal_sino, metal_mask, theta = mar_utils.estimate_metal_sino(ct_model, sino, init_recon, metal_threshold=0.1)
    plastic_sino = sino - metal_sino
    mbirjax.slice_viewer(metal_sino, plastic_sino, slice_axis=0, slice_label='Metal Sino', slice_label2='Plastic Sino', title='Views')

    print("\n*******************************************************",
          "\n************ Calculate MAR sinogram weights ***********",
          "\n*******************************************************")
    weights_mar = ct_model.gen_weights_mar(sino, init_recon=init_recon, beta=1.0, gamma=3.0)

    print("\n*******************************************************",
          "\n*********** Perform recon with MAR weights. ***********",
          "\n*******************************************************")
    # ##########################
    # Perform VCD reconstruction
    time0 = time.time()
    recon_plastic, recon_params = ct_model.recon(plastic_sino, weights=weights_mar, init_recon=init_recon)
    recon_plastic.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon with MAR weight is {:.3f} seconds'.format(elapsed))
    # ##########################

    # #### combine metal and plastic recons
    recon_mar = recon_plastic*(1.0-metal_mask) + init_recon*metal_mask

    # #### Display results
    # change the image data shape to (slices, rows, cols)
    init_recon = np.transpose(init_recon, axes=(2, 0, 1))
    recon_mar = np.transpose(recon_mar, axes=(2, 0, 1))

    vmin = 0
    vmax = downsample_factor[0] * 0.025
    mbirjax.slice_viewer(init_recon, recon_mar, vmin=0, vmax=vmax, slice_axis=0, slice_label='MBIR', slice_label2='MBIR MAR', title='Axial Slice')
    mbirjax.slice_viewer(init_recon, recon_mar, vmin=0, vmax=vmax, slice_axis=1, slice_label='MBIR', slice_label2='MBIR MAR', title='Coronal Slice')
    mbirjax.slice_viewer(init_recon, recon_mar, vmin=0, vmax=vmax, slice_axis=2, slice_label='MBIR', slice_label2='MBIR MAR', title='Sagittal Slice')
