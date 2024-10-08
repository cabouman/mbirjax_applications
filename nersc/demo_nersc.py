"""
NERSC Sand Demo.

This script demonstrates a basic workflow for running MBIRJAX reconstructions on NERSC datasets.
"""

import numpy as np
import time
import jax.numpy as jnp
import mbirjax
import nersc_utils
import h5py

# Data download and extraction
# An example NERSC dataset (tarball) will be downloaded from `dataset_url`, and saved to `download_dir`.
dataset_url = 'https://engineering.purdue.edu/~bouman/data_repository/data/nersc-sand.tgz'
download_dir = './demo_data/'
_, dataset_path = nersc_utils.download_and_extract_tar(dataset_url, download_dir)

# Get parameters from data. 
with h5py.File(dataset_path, "r") as data: 
    numslices = int(data['/measurement/instrument/detector/dimension_y'][0])
    numrays = int(data['/measurement/instrument/detector/dimension_x'][0])
    numangles = int(data['/process/acquisition/rotation/num_angles'][0])
    pxsize = data['/measurement/instrument/detector/pixel_size'][0] / 10.0  # /10 to convert units from mm to cm
    propagation_dist = data['/measurement/instrument/camera_motor_stack/setup/camera_distance'][1]
    kev = data['/measurement/instrument/monochromator/energy'][0] / 1000
    angularrange = data['/process/acquisition/rotation/range'][0]
    angles = np.deg2rad(data['exchange/theta'])  # MBIRJAX uses radians

# Print these parameters
print(
    f"{dataset_path}: \
        \n\t * slices: {numslices}, rays: {numrays}, angles: {numangles}, angularrange: {angularrange},\
        \n\t * pxsize: {pxsize*10000:.3f} um, distance: {propagation_dist:.3f} mm. energy: {kev} keV",
        end = "\n\n"
        )

# This code selects a subset of slices centered around the middle slice of the sinogram, ... 
# ... defined by num_slices_subset, and returns the starting and ending slice indices as a tuple.
num_slices_length = 10  # Example value, you can adjust this
mid_slice = numslices // 2
sino_used = (mid_slice - num_slices_length // 2, mid_slice + num_slices_length // 2)
# Define sino_used = (0, num_slices) for full recon

# Get (subsetted) object, blank, and dark scans
with h5py.File(dataset_path, 'r') as data:
    obj_scan = data['exchange/data'][:, sino_used[0]:sino_used[1], :] 
    blank_scan = data['exchange/data_white'][:, sino_used[0]:sino_used[1], :] 
    dark_scan = data['exchange/data_dark'][:, sino_used[0]:sino_used[1], :] 

angles = -angles # I don't know the reason behind the angle flip. Maybe the rotation direction is defined differently in MBIRJAX and in LBNL instrumentation?
obj_scan = obj_scan.astype(np.float32,copy=False)
blank_scan = blank_scan.astype(np.float32,copy=False)
dark_scan = dark_scan.astype(np.float32,copy=False)
print("shape of object scan = ", obj_scan.shape)
print("shape of blank scan = ", blank_scan.shape)
print("shape of dark scan = ", dark_scan.shape)
        
# Recon parameters
cor = 1265.5  # this is used to calculated det_channel_offset. det_channel_offset = int((cor - numrays/2)
sharpness = 0.0
recon_margin = 256 # margin width of the reconstruction. The reconstruction region is zero-padded and cropped afterward.

print("Computing sinogram data from object, blank, and dark scans ...")
sinogram, _ = mbirjax.preprocess.utilities.compute_sino_transmission(obj_scan, blank_scan, dark_scan)
print("shape of sinogram data = ", sinogram.shape) 
det_channel_offset = int((cor - numrays/2))
sinogram = jnp.array(sinogram) # converts FROM a numpy ndarray
# View sinogram
mbirjax.slice_viewer(sinogram.transpose((0, 2, 1)), title='Original sinogram')

print("Set up MBIRJAX model", end="\n\n")
# Initialize model
parallel_model = mbirjax.ParallelBeamModel(sinogram_shape=sinogram.shape, angles=angles)
recon_shape = parallel_model.get_params("recon_shape")
recon_shape = (recon_shape[0] + recon_margin*2, recon_shape[1] + recon_margin*2, recon_shape[2])
parallel_model.set_params(recon_shape=recon_shape)

# Generate weights array (leave commented out for now)
# weights = parallel_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

# Set reconstruction parameter values
parallel_model.set_params(sharpness=sharpness, det_channel_offset=det_channel_offset, verbose=1)
# Print out model parameters
parallel_model.print_params()

print("Perform MBIRJAX reconstruction", end="\n\n")

# Perform VCD reconstruction
time0 = time.time()
# default number of iterations for recon is 13
# recon, recon_params = parallel_model.recon(sinogram, weights=weights)
recon, _ = parallel_model.recon(sinogram)
recon.block_until_ready()
elapsed = time.time() - time0
print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))

# Display recon results
recon /= pxsize  # scale by pixel size to units of 1/cm
recon = recon[recon_margin:-recon_margin, recon_margin:-recon_margin, :]  # undo padding of recon   
# Masking the reconstruction to display the circular ROR region
circular_mask = nersc_utils.create_circular_mask(recon.shape[0], recon.shape[1]).astype(int)
recon = recon*circular_mask[:, :, np.newaxis]
mbirjax.slice_viewer(recon, title='VCD Recon')
