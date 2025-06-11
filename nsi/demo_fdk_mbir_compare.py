import numpy as np
import os
import time
import pprint
import jax
import jax.numpy as jnp
import jax.lax as lax


import mbirjax as mj
import mbirjax.preprocess as mjp
import mar_utils
import pprint

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('This script is demonstrates the preprocessing and reconstruction of NSI an dataset\
    \n\t using both FDK and MBIR reconstruction.\n')

    # User defined params
    output_path = './output/nsi_demo/'  # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

    # path to store and extract the NSI data and metadata.
    download_dir = './demo_data/'

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

    # Download and extract data. Then set path to NSI scan directory.
    dataset_dir = mj.download_and_extract_tar(dataset_url, download_dir)

    # preprocessing parameters
    downsample_factor = [4, 4]  # downsample factor of scan view images along detector rows and detector columns.
    subsample_view_factor = 8  # view subsample factor.

    # recon parameters
    sharpness = 1.0
    snr_db = 30.0

    print("\n*******************************************************",
          "\n************** NSI dataset preprocessing **************",
          "\n*******************************************************")
    sino, cone_beam_params, optional_params = \
        mjp.nsi.compute_sino_and_params(dataset_dir,
                                                       downsample_factor=downsample_factor,
                                                       subsample_view_factor=subsample_view_factor)

    print("\n*******************************************************",
          "\n***************** Set up MBIRJAX model ****************",
          "\n*******************************************************")
    # Construct cone beam object using NSI parameters
    ct_model = mj.ConeBeamModel(**cone_beam_params)

    # Set optional NSI geometry parameters
    ct_model.set_params(**optional_params)

    # Set user determined parameter values
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
    #mj.slice_viewer(fdk_recon)

    print("\n*******************************************************",
          "\n************** Perform MBIR reconstruction ************",
          "\n*******************************************************")

    # #### Perform MBIR reconstruction
    time0 = time.time()
    mbir_recon, mbir_recon_params = ct_model.recon(sino, weights=weights)
    mbir_recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))

    # #### Print out parameters used in recon
    pprint.pprint(mbir_recon_params._asdict())

    # #### Save MBIR reconstruction to HDF5 file output
    ct_model.save_recon_dict_to_hdf5(os.path.join(output_path, "recon.h5"), mbir_recon)

    # #### Display results
    # change the image data shape to (slices, rows, cols)
    fdk_recon = np.transpose(fdk_recon, axes=(2, 0, 1))
    mbir_recon = np.transpose(mbir_recon, axes=(2, 0, 1))

    # Display FDK versus MBIR
    vmin = 0
    vmax = downsample_factor[0] * 0.025
    mj.slice_viewer(fdk_recon, mbir_recon, vmin=0, vmax=vmax, slice_label= ["FDK Recon", "MBIR Recon"], title='Axial Slice')
