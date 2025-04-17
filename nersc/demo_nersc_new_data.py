"""
NERSC Fuel Cell and Simulated Permafrost Demo.

This script demonstrates a basic workflow for running MBIRJAX reconstructions on NERSC fuel cell and simulated permafrost datasets.
"""

import numpy as np
import time
import jax.numpy as jnp
import mbirjax
import nersc_utils_update
import stripe_removal
import stripe_removal_jax
import h5py
import os

# Set the GPU memory fraction for JAX
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.98'

# Choose the dataset
dataset = "fuelcell" # "fuelcell" or "permafrost"

# Data download and extraction
# A NERSC dataset (fuelcell or permafrost) will be downloaded from `dataset_url`, and saved to `download_dir`.
if dataset == "fuelcell":
    dataset_url = 'https://drive.google.com/uc?id=1L_fUFBlhDzJfCNoAt-iNHMisqyb8Cxcx'
else:
    dataset_url = 'https://drive.google.com/uc?id=1TKcznoY-guNbKyY1Ql-Jo1QpgYfVlCkN'
download_dir = './demo_data/'
dataset_path = nersc_utils_update.download_file(dataset_url, download_dir)

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
        \n\t * pxsize: {pxsize * 10000:.3f} um, distance: {propagation_dist:.3f} mm. energy: {kev} keV",
    end="\n\n"
)

# This code selects a subset of slices centered around the middle slice of the sinogram, ...
# ... defined by num_slices_subset, and returns the starting and ending slice indices as a tuple.
num_slices_length = 10  # Example value, you can adjust this
mid_slice = numslices // 2
if dataset == "fuelcell":
    sino_used = (149, 159)
else:
    sino_used = (mid_slice - num_slices_length // 2, mid_slice + num_slices_length // 2)
# Define sino_used = (0, num_slices) for full recon

# Get (subsetted) object, blank, and dark scans
with h5py.File(dataset_path, 'r') as data:
    obj_scan = data['exchange/data'][:, sino_used[0]:sino_used[1], :]
    blank_scan = data['exchange/data_white'][:, sino_used[0]:sino_used[1], :]
    dark_scan = data['exchange/data_dark'][:, sino_used[0]:sino_used[1], :]

angles = -angles  # I don't know the reason behind the angle flip. Maybe the rotation direction is defined differently in MBIRJAX and in LBNL instrumentation?
obj_scan = obj_scan.astype(np.float32, copy=False)
blank_scan = blank_scan.astype(np.float32, copy=False)
dark_scan = dark_scan.astype(np.float32, copy=False)
print("shape of object scan = ", obj_scan.shape)
print("shape of blank scan = ", blank_scan.shape)
print("shape of dark scan = ", dark_scan.shape)

# Recon parameters
if dataset == "fuelcell":
    cor = 1283.0 # this is used to calculated det_channel_offset. det_channel_offset = int((cor - numrays/2)
else:
    cor = 1208.875
print("center of rotation = ", cor)
sharpness = 1.0

print("Computing sinogram data from object, blank, and dark scans ...")
sinogram, _ = mbirjax.preprocess.utilities.compute_sino_transmission(obj_scan, blank_scan, dark_scan)
print("shape of sinogram data = ", sinogram.shape)
det_channel_offset = int((cor - numrays / 2))

# CPU-based stripe removal
# sinogram = stripe_removal.remove_all_stripe(sinogram) # NumPy-based local implementation of tomopy_remove_all_stripe()
# sinogram = stripe_removal.remove_stripe_fw(sinogram) # NumPy-based local implementation of tomopy.remove_stripe_fw()

# GPU-based stripe removal
sinogram = jnp.array(sinogram)  # converts FROM a numpy ndarray
# sinogram = stripe_removal_jax.remove_all_stripe_jax(sinogram) # JAX-based local implementation of tomopy_remove_all_stripe()
# sinogram = stripe_removal_jax.remove_stripe_fw_jax(sinogram) # JAX-based local implementation of tomopy.remove_stripe_fw()

# View sinogram
if dataset == "fuelcell":
    mbirjax.slice_viewer(sinogram.transpose((0, 2, 1)), vmin = 0.56, vmax = 2.17, title='Original sinogram')
else:
    mbirjax.slice_viewer(sinogram.transpose((0, 2, 1)), vmin = 0.38, vmax = 0.90, title='Original sinogram')

print("Set up MBIRJAX model", end="\n\n")
# Initialize model
parallel_model = mbirjax.ParallelBeamModel(sinogram_shape=sinogram.shape, angles=angles)
recon_shape = parallel_model.get_params("recon_shape")
recon_row_scale = 1.2 # Try to increase this value if there is bright rings around the boundary of the reconstruction
recon_col_scale = 1.2 # Try to increase this value if there is bright rings around the boundary of the reconstruction
parallel_model.scale_recon_shape(row_scale=recon_row_scale, col_scale=recon_col_scale)

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
# Undo padding of recon
center_x, center_y = recon.shape[0] // 2, recon.shape[1] // 2
crop_size = numrays // 2
recon = recon[center_y - crop_size:center_y + crop_size, center_x - crop_size:center_x + crop_size]
print("shape of recon = ", recon.shape)
# Masking the reconstruction to display the circular ROR region
circular_mask = nersc_utils_update.create_circular_mask(recon.shape[0], recon.shape[1]).astype(int)
recon = recon * circular_mask[:, :, np.newaxis]
mbirjax.slice_viewer(recon, vmin = 0, vmax = 10, title='VCD Recon for {} data'.format(dataset))