import numpy as np
import os
import time
import jax.numpy as jnp
import mbirjax
import mbirjax
import mbirjax.plot_utils as pu
import nersc_utils
import h5py

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
   
    # ###################### geometry parameters
    cor = 1265.5 # this is used to calculated det_channel_offset. det_channel_offset = int((cor - numrays/2)
    # ###################### recon parameters
    sharpness = 0.0
    recon_margin = 256 # margin width of the reconstruction. The region of reconstruction will zero-pad during the reconstruction process. This margin will be cropped out afterwards.
    # ###################### End of parameters

    print("\n*******************************************************",
          "\n**************** Load NERSC meta data *****************",
          "\n*******************************************************")
    
    with h5py.File(dataset_path, 'r') as f:
        numslices = int(f['/measurement/instrument/detector/dimension_y'][0])
        numrays = int(f['/measurement/instrument/detector/dimension_x'][0])
        pxsize = f['/measurement/instrument/detector/pixel_size'][0] / 10.0  # /10 to convert units from mm to cm
        numangles = int(f['/process/acquisition/rotation/num_angles'][0])
        propagation_dist = f['/measurement/instrument/camera_motor_stack/setup/camera_distance'][1]
        kev = f['/measurement/instrument/monochromator/energy'][0] / 1000
        angularrange = f['/process/acquisition/rotation/range'][0]

    print(f"{dataset_path}: \
            \n\t * slices: {numslices}, rays: {numrays}, angles: {numangles}, angularrange: {angularrange},\
            \n\t * pxsize: {pxsize*10000:.3f} um, distance: {propagation_dist:.3f} mm. energy: {kev} keV")
    if kev>100:
        print('white light mode detected; energy is set to 30 kev for the phase retrieval function')

    print("\n*******************************************************",
          "\n************* Preprocess NERSC data ****************",
          "\n*******************************************************")
    
    sinoused = (-1,10,1) # using the whole numslices will make it run out of memory
    if sinoused[0] < 0:
        sinoused = (int(np.floor(numslices / 2.0) - np.ceil(sinoused[1] / 2.0)), int(np.floor(numslices / 2.0) + np.floor(sinoused[1] / 2.0)), 1)

    print("Reading object, blank, and dark scans from the input hdf5 file ...")
    with h5py.File(dataset_path, 'r') as f:
        obj_scan = f['exchange/data'][:, sinoused[0]:sinoused[1], :]  # sinoused[0]:sinoused[1] effectively subsets the slices
        blank_scan = f['exchange/data_white'][:, sinoused[0]:sinoused[1], :] 
        dark_scan = f['exchange/data_dark'][:, sinoused[0]:sinoused[1], :] 
        angles = np.deg2rad(f['exchange/theta']) 

    angles = -angles # I don't know the reason behind the angle flip. Maybe the rotation direction is defined differently in MBIRJAX and in LBNL instrumentation?
    obj_scan = obj_scan.astype(np.float32,copy=False)
    blank_scan = blank_scan.astype(np.float32,copy=False)
    dark_scan = dark_scan.astype(np.float32,copy=False)
    print("shape of object scan = ", obj_scan.shape)
    print("shape of blank scan = ", blank_scan.shape)
    print("shape of dark scan = ", dark_scan.shape)
    
    print("Computing sinogram data from object, blank, and dark scans ...")
    sinogram, _ = mbirjax.preprocess.utilities.compute_sino_transmission(obj_scan, blank_scan, dark_scan)
    print("shape of sinogram data = ", sinogram.shape) 

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
    # default number of iterations for recon is 13
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
    recon = recon[recon_margin:-recon_margin, recon_margin:-recon_margin, :] # undo padding of recon   

    print("Masking the reconstruction to display the circular ROR region ...")
    circular_mask = nersc_utils.create_circular_mask(recon.shape[0],recon.shape[1]).astype(int)
    recon = recon*circular_mask[:,:,np.newaxis]
    pu.slice_viewer(recon, vmin=-5, vmax=10, title='VCD Recon (right)')
