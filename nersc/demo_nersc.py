import numpy as np
import os
import time
import jax.numpy as jnp
import mbirjax
import dxchange
import tomopy
import mbirjax
import mbirjax.plot_utils as pu
import nersc_utils

import pprint
pp = pprint.PrettyPrinter(indent=4)


if __name__ == "__main__":
    # ##################### User defined params. Change the parameters below for your own use case.
    output_path = './output/nersc_demo_sand/'  # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist
    
    # ##### params for dataset downloading. User may change these parameters for their own datasets.
    # An example NERSC dataset (tarball) will be downloaded from `dataset_url`, and saved to `download_dir`.
    # url to NERSC dataset. 
    dataset_url = 'https://engineering.purdue.edu/~bouman/data_repository/data/nersc-sand.tgz'
    # destination path to download and extract the NERSC data and metadata.
    download_dir = './demo_data/'
    _, dataset_path = nersc_utils.download_and_extract_tar(dataset_url, download_dir)
   
    # ##################### preprocessing parameters
    # #### paramters specific to outlier removal
    outlier_diff1D = 750
    outlier_size1D = 3
 
    # #### paramters specific to projection data smoothing
    ringSize = 5 #Size of the filter used to smooth the projection data 
    
    # #### parameters specific to strip artifact removal
    ringVo_snr= 1.1 #Ring Removal SNR: Sensitivity of large stripe detection method. Smaller is more sensitive. No affect on small stripes. Recommended values: 1.1 -- 3.0.
    ringVo_la_size=75 #Large Ring Size: Window size of the median filter to remove large stripes. Set to appx width of large stripes -- should be larger value than Small Ring Size. Always choose odd value, set to 1 to turn off.
    ringVo_sm_size=15 #Small Ring Size: Window size of the median filter to remove small stripes. Larger is stronger but takes longer. Set to appx width of small stripes. Always choose odd value, set to 1 to turn off.
    ringVo_dim=1

    # ###################### geometry parameters
    cor = 1265.5 # this is used to calculated det_channel_offset. det_channel_offset = int((cor - numrays/2)
    
    # ###################### recon parameters
    sharpness = 0.0
    recon_margin = 256 # margin width of the reconstruction. The region of reconstruction will zero-padded during the reconstruction process. This margin will be cropped out afterwards.
    # ###################### End of parameters

    print("\n*******************************************************",
          "\n**************** Load NERSC meta data *****************",
          "\n*******************************************************")
    numslices = int(dxchange.read_hdf5(dataset_path, "/measurement/instrument/detector/dimension_y")[0])
    numrays = int(dxchange.read_hdf5(dataset_path, "/measurement/instrument/detector/dimension_x")[0])
    pxsize = dxchange.read_hdf5(dataset_path, "/measurement/instrument/detector/pixel_size")[0] / 10.0  # /10 to convert units from mm to cm
    numangles = int(dxchange.read_hdf5(dataset_path, "/process/acquisition/rotation/num_angles")[0])
    propagation_dist = dxchange.read_hdf5(dataset_path, "/measurement/instrument/camera_motor_stack/setup/camera_distance")[1]
    kev = dxchange.read_hdf5(dataset_path, "/measurement/instrument/monochromator/energy")[0] / 1000
    angularrange = dxchange.read_hdf5(dataset_path, "/process/acquisition/rotation/range")[0]

    print(f"{dataset_path}: \
            \n\t * slices: {numslices}, rays: {numrays}, angles: {numangles}, angularrange: {angularrange},\
            \n\t * pxsize: {pxsize*10000:.3f} um, distance: {propagation_dist:.3f} mm. energy: {kev} keV")
    if kev>100:
        print('white light mode detected; energy is set to 30 kev for the phase retrieval function')

    print("\n*******************************************************",
          "\n************* Preprocess NERSC data ****************",
          "\n*******************************************************")
    sinoused = (-1,10,1) #using the whole numslices will make it run out of memory
    if sinoused[0] < 0:
        sinoused = (int(np.floor(numslices / 2.0) - np.ceil(sinoused[1] / 2.0)), int(np.floor(numslices / 2.0) + np.floor(sinoused[1] / 2.0)), 1)

    print("Reading object, blank, and dark scans from the input hdf5 file ...")
    obj_scan, blank_scan, dark_scan, angles = dxchange.exchange.read_aps_tomoscan_hdf5(dataset_path, sino=(sinoused[0],sinoused[1],sinoused[2]))
    angles = -angles # I don't know the reason behind the angle flip. Maybe the rotation direction is defined differently in MBIRJAX and in LBNL instrumentation?
    obj_scan = obj_scan.astype(np.float32,copy=False)
    blank_scan = blank_scan.astype(np.float32,copy=False)
    dark_scan = dark_scan.astype(np.float32,copy=False)
    print("shape of object scan = ", obj_scan.shape)
    print("shape of blank scan = ", blank_scan.shape)
    print("shape of dark scan = ", dark_scan.shape)
    
    print("Removing high intensity bright spots from object and blank scans ...")
    tomopy.misc.corr.remove_outlier(obj_scan, outlier_diff1D, size=outlier_size1D, out=obj_scan, ncore=64)
    tomopy.misc.corr.remove_outlier(blank_scan, outlier_diff1D, size=outlier_size1D, out=blank_scan, ncore=64)
    
    print("Computing sinogram data from object, blank, and dark scans ...")
    sinogram, _ = mbirjax.preprocess.utilities.compute_sino_transmission(obj_scan, blank_scan, dark_scan)
    print("shape of sinogram data = ", sinogram.shape) 

    print("Normalizing sinogram data using a smoothing filter ...")
    sinogram = tomopy.remove_stripe_sf(sinogram, size=ringSize)
    
    print("Removing all types of stripe artifacts from sinogram using Nghia Vo’s approach ...")
    sinogram = tomopy.remove_all_stripe(sinogram,snr=ringVo_snr, la_size=ringVo_la_size, sm_size=ringVo_sm_size, dim=ringVo_dim)

    det_channel_offset = int((cor - numrays/2))

    sinogram = jnp.array(sinogram)

    # View sinogram
    pu.slice_viewer(sinogram.transpose((0, 2, 1)), title='Original sinogram')

    print("\n*******************************************************",
          "\n***************** Set up MBIRJAX model ****************",
          "\n*******************************************************")
    parallel_model = mbirjax.ParallelBeamModel(sinogram_shape=sinogram.shape, angles=angles)
    recon_shape = parallel_model.get_params('recon_shape')
    recon_shape = (recon_shape[0] + recon_margin*2, recon_shape[1] + recon_margin*2, recon_shape[2])
    parallel_model.set_params(recon_shape=recon_shape)

    # Generate weights array
    # weights = parallel_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    parallel_model.set_params(sharpness=sharpness, det_channel_offset=det_channel_offset, verbose=1)


    # Print out model parameters
    parallel_model.print_params()
    
    print("\n*******************************************************",
          "\n*********** Perform MBIRJAX reconstruction ************",
          "\n*******************************************************")
    # ##########################
    # Perform VCD reconstruction
    time0 = time.time()
    #default number of iterations for recon is 13
    # recon, recon_params = parallel_model.recon(sinogram, weights=weights)
    recon, recon_params = parallel_model.recon(sinogram)

    recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))
    # ##########################

    print("\n*******************************************************",
          "\n**************** Display recon results ****************",
          "\n*******************************************************")
    print("Scaling the reconstruction by pixel size ...")
    recon /= pxsize #scale by pixel size to units of 1/cm
    
    print("Cropping the reconstruction margin ...")
    recon = recon[recon_margin:-recon_margin, recon_margin:-recon_margin, :] #undo padding of recon   

    print("Masking the reconstruction to display the circular ROR region ...")
    circular_mask = nersc_utils.create_circular_mask(recon.shape[0],recon.shape[1]).astype(int)
    recon = recon*circular_mask[:,:,np.newaxis]
    pu.slice_viewer(recon, vmin=-5, vmax=10, title='VCD Recon (right)')
