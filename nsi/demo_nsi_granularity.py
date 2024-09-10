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
    dataset_url = 'https://engineering.purdue.edu/~bouman/data_repository/data/demo_data_nsi.tgz'
    # destination path to download and extract the NSI data and metadata.
    download_dir = './demo_data/'
    # Path to NSI scan directory.
    _, dataset_dir = demo_utils.download_and_extract_tar(dataset_url, download_dir)
    # for testing user prompt in NSI preprocessing function
    # dataset_dir = "/depot/bouman/data/share_conebeam_data/Autoinjection-Full-LowRes/Vertical-0.5mmTin"

    # #### preprocessing parameters
    downsample_factor = [4, 4]  # downsample factor of scan images along detector rows and detector columns.
    subsample_view_factor = 1  # view subsample factor.

    # #### recon parameters
    sharpness = 1.5
    snr_db = 35
    granularity = [1, 3, 9, 27, 81, 243]
    partition_sequence = [0, 1, 2, 3, 4, 5]
    num_iterations = 15
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

    # Set granularity and partition sequence
    ct_model.set_params(granularity=granularity)
    ct_model.set_params(partition_sequence=partition_sequence)

    # Print out model parameters
    ct_model.print_params()

    print("\n*******************************************************",
          "\n************** Calculate sinogram weights *************",
          "\n*******************************************************")
    weights = ct_model.gen_weights(sino, weight_type='transmission_root')

    print("\n*******************************************************",
          "\n************** Perform VCD reconstruction *************",
          "\n*******************************************************")
    # ##########################
    # Perform VCD reconstruction
    time0 = time.time()

    recon, recon_params = ct_model.recon(sino, weights=weights, num_iterations=num_iterations, compute_prior_loss=True)

    recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))
    # ##########################

    # Print out parameters used in recon
    pprint.pprint(recon_params._asdict())

    mbirjax.preprocess.export_recon_to_hdf5(recon, os.path.join(output_path, "recon.h5"),
                                            recon_description="MBIRJAX recon of MAR phantom",
                                            alu_description="1 ALU = 0.508 mm")

    # change the image data shape to (slices, rows, cols), so that the rotation axis points up when viewing the
    # coronal/sagittal slices with slice_viewer
    recon = np.transpose(recon, (2, 1, 0))
    recon = recon[:, :, ::-1]

    vmin = 0
    vmax = downsample_factor[0] * 0.008
    # Display results
    mbirjax.slice_viewer(recon, vmin=0, vmax=vmax, slice_axis=0, slice_label='Axial Slice', title='MBIRJAX recon')
    mbirjax.slice_viewer(recon, vmin=0, vmax=vmax, slice_axis=1, slice_label='Coronal Slice', title='MBIRJAX recon')
    mbirjax.slice_viewer(recon, vmin=0, vmax=vmax, slice_axis=2, slice_label='Sagittal Slice', title='MBIRJAX recon')
