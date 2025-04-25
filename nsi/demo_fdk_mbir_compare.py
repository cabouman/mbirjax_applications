import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import mbirjax
import demo_utils
import pprint

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('This script is a demonstration of the preprocessing module of NSI dataset. Demo functionality includes:\
    \n\t * downloading NSI dataset from specified urls;\
    \n\t * Loading object scans, blank scan, dark scan, view angles, and MBIRJAX geometry parameters;\
    \n\t * Computing sinogram from object scan, blank scan, and dark scan images;\
    \n\t * Computing a 3D reconstruction from the sinogram using MBIRJAX;\
    \n\t * Displaying the results.\n')

    # ###################### User defined params. Change the parameters below for your own use case.
    output_path = './output/nsi_demo/'  # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

    # ##### params for dataset downloading. User may change these parameters for their own datasets.
    # An example NSI dataset (tarball) will be downloaded from `dataset_url`, and saved to `download_dir`.
    # url to NSI dataset.
    # dataset_url = 'https://www.datadepot.rcac.purdue.edu/bouman/data/demo_data_nsi.tgz'
    dataset_url = 'https://www.datadepot.rcac.purdue.edu/bouman/data/demo_nsi_vert_no_metal_all_views.tgz'
    # destination path to download and extract the NSI data and metadata.
    download_dir = './demo_data/'
    # Path to NSI scan directory.
    _, dataset_dir = demo_utils.download_and_extract_tar(dataset_url, download_dir)
    # for testing user prompt in NSI preprocessing function
    # dataset_dir = "/depot/bouman/data/share_conebeam_data/Autoinjection-Full-LowRes/Vertical-0.5mmTin"

    # #### preprocessing parameters
    downsample_factor = [4, 4]  # downsample factor of scan images along detector rows and detector columns.
    subsample_view_factor = 4  # view subsample factor.

    # #### recon parameters
    sharpness = 1.0
    snr_db = 30.0
    # ###################### End of parameters

    print("\n*******************************************************",
          "\n************** NSI dataset preprocessing **************",
          "\n*******************************************************")
    sino, cone_beam_params, optional_params = \
        mbirjax.preprocess.nsi.compute_sino_and_params(dataset_dir,
                                                       downsample_factor=downsample_factor,
                                                       subsample_view_factor=subsample_view_factor)

    print("\n*******************************************************",
          "\n***************** Set up MBIRJAX model ****************",
          "\n*******************************************************")
    # ConeBeamModel constructor
    ct_model = mbirjax.ConeBeamModel(**cone_beam_params)

    # Set additional geometry arguments
    ct_model.set_params(**optional_params)

    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, snr_db=snr_db, verbose=1)

    # Print out model parameters
    ct_model.print_params()

    print("\n*******************************************************",
          "\n************** Calculate sinogram weights *************",
          "\n*******************************************************")
    weights = ct_model.gen_weights(sino, weight_type='transmission_root')

    print("\n******************************************************",
          "\n************** Perform FDK reconstruction ************",
          "\n******************************************************")

    # ##########################
    # Perform FDK reconstruction
    fdk_recon = ct_model.direct_recon(sino)
    #mbirjax.slice_viewer(fdk_recon)

    print("\n*******************************************************",
          "\n************** Perform MBIR reconstruction ************",
          "\n*******************************************************")

    # ##########################
    # Perform MBIR reconstruction
    time0 = time.time()
    mbir_recon, mbir_recon_params = ct_model.recon(sino, weights=weights)
    mbir_recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))
    # ##########################

    # Print out parameters used in recon
    pprint.pprint(mbir_recon_params._asdict())

    mbirjax.preprocess.export_recon_to_hdf5(mbir_recon, os.path.join(output_path, "recon.h5"),
                                            recon_description="MBIRJAX recon of phantom",
                                            alu_description="1 ALU = 0.508 mm")

    # Display results
    vmin = 0
    vmax = downsample_factor[0] * 0.008
    mbirjax.slice_viewer(fdk_recon, data2=mbir_recon, vmin=0, vmax=vmax, slice_axis=2, slice_axis2=2, slice_label='FDK', slice_label2='MBIR', title='Axial Slice')
    mbirjax.slice_viewer(fdk_recon, data2=mbir_recon, vmin=0, vmax=vmax, slice_axis=0, slice_axis2=0, slice_label='FDK', slice_label2='MBIR', title='Coronal Slice')
    mbirjax.slice_viewer(fdk_recon, data2=mbir_recon, vmin=0, vmax=vmax, slice_axis=1, slice_axis2=1, slice_label='FDK', slice_label2='MBIR', title='Sagittal Slice')
