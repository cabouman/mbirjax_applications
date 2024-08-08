import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import mbirjax
import dxchange
import tomopy
import mbirjax
import mbirjax.plot_utils as pu

pp = pprint.PrettyPrinter(indent=4)

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

if __name__ == "__main__":
    # ###################### User defined params. Change the parameters below for your own use case.
    output_path = './output/nserc_demo_sand/'  # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

    dataset_path = "./20240425_164409_nist-sand-30-200-mix_27keV_z8mm_n657.h5"    
    
    # #### recon parameters
    sharpness = 0.0
    # ###################### End of parameters

    print("\n*******************************************************",
          "\n*********** Load NSERC data and meta data *************",
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
          "\n************* Preprocessing NSERC data ****************",
          "\n*******************************************************")
    sinoused = (-1,10,1) #using the whole numslices will make it run out of memory
    if sinoused[0] < 0:
            sinoused = (int(np.floor(numslices / 2.0) - np.ceil(sinoused[1] / 2.0)), int(np.floor(numslices / 2.0) + np.floor(sinoused[1] / 2.0)), 1)

    tomo, flat, dark, anglelist = dxchange.exchange.read_aps_tomoscan_hdf5(dataset_path, sino=(sinoused[0],sinoused[1],sinoused[2]))
    anglelist = -anglelist
    tomo = tomo.astype(np.float32,copy=False)
    flat = flat.astype(np.float32,copy=False)
    dark = dark.astype(np.float32,copy=False)
    print("shape of tomo = ", tomo.shape)
    print("shape of flat = ", flat.shape)
    print("shape of dark = ", dark.shape)
    
    outlier_diff1D = 750
    outlier_size1D = 3
    #remove_outlier1d(tomo, outlier_diff1D, size=outlier_size1D, out=tomo, ncore=64)
    #remove_outlier1d(flat, outlier_diff1D, size=outlier_size1D, out=flat, ncore=64)

    tomopy.normalize(tomo, flat, dark, out=tomo, ncore=64)
    tomopy.minus_log(tomo, out=tomo, ncore=64);

    # ringSize = 5
    # tomo = tomopy.remove_stripe_sf(tomo, size=ringSize)

    ringVo_snr= 1.1 #Ring Removal SNR: Sensitivity of large stripe detection method. Smaller is more sensitive. No affect on small stripes. Recommended values: 1.1 -- 3.0.
    ringVo_la_size=75 #Large Ring Size: Window size of the median filter to remove large stripes. Set to appx width of large stripes -- should be larger value than Small Ring Size. Always choose odd value, set to 1 to turn off.
    ringVo_sm_size=15 #Small Ring Size: Window size of the median filter to remove small stripes. Larger is stronger but takes longer. Set to appx width of small stripes. Always choose odd value, set to 1 to turn off.
    ringVo_dim=1

    tomo = tomopy.remove_all_stripe(tomo,snr=ringVo_snr, la_size=ringVo_la_size, sm_size=ringVo_sm_size, dim=ringVo_dim, ncore=64)

    cor = 1265.5
    theshift = int((cor - numrays/2))

    sinogram = jnp.array(tomo)

    # = jax.device_put() 
    # View sinogram
    pu.slice_viewer(tomo.transpose((0, 2, 1)), title='Original sinogram')


    print("\n*******************************************************",
          "\n*********** Perform MBIRJAX reconstruction ************",
          "\n*******************************************************")
    parallel_model = mbirjax.ParallelBeamModel(sinogram_shape=sinogram.shape, angles=anglelist)
    recon_shape = parallel_model.get_params('recon_shape')
    npad = 256
    recon_shape = (recon_shape[0] + npad*2, recon_shape[1] + npad*2, recon_shape[2])
    parallel_model.set_params(recon_shape=recon_shape)

    # Generate weights array
    # weights = parallel_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    parallel_model.set_params(sharpness=sharpness, det_channel_offset=theshift, verbose=1)


    # Print out model parameters
    parallel_model.print_params()

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
    recon /= pxsize #scale by pixel size to units of 1/cm
    recon = recon[npad:-npad, npad:-npad, :] #undo padding of recon   

    themask = create_circular_mask(recon.shape[0],recon.shape[1]).astype(int)
    recon = recon*themask[:,:,np.newaxis]
    pu.slice_viewer(recon, vmin=-5, vmax=10, title='VCD Recon (right)')
