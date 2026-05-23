"""
Hyperspectral Neutron Tomography
--------------------------------

This script demonstrates hyperspectral neutron data preprocessing and fast hyperspectral reconstruction for Ni-Cu-Al dataset
"""

import os
import shutil
import numpy as np
import mbirjax as mj
import hsnt_prep_utils as h_preproc
import matplotlib.pyplot as plt

# Setup paths
base_path = '/depot/bouman/data/ORNL/hsnt/tci_2025_Ni_Cu_Al'
ob_folder_path = os.path.join(base_path, 'open_beam')  # Raw open-beam folder path, may contain one or more observations
proj_folder_path = os.path.join(base_path, 'projections')  # Raw projection folder path, may contain one or more views
dataset_name = 'Ni_Cu_Al_dataset'
output_file_name = 'dehydrated_recons_' + dataset_name + '.h5'  # Output folder name

# Setup parameters
wave_idx_start = 100  # Index of the 1st wavelength bin to be loaded
num_total_wave = 1200  # Number of total wavelength bins to be loaded
angles = [0.0, 6.2, 12.399, 16.231, 22.43, 32.461, 38.661, 42.492, 48.692, 58.723,
          64.922, 74.953, 81.153, 84.984, 91.184, 101.215, 107.415, 111.246, 117.446,
          127.477, 133.676, 143.707, 149.907, 153.738, 159.938, 169.969, 176.168]  # View angles in degrees
num_materials = 3  # Number of materials in the sample
recon_snr_db = 25  # Assumed SNR for the dataset in dB
verbose = 0  # Print nothing if 0

# Setup background calibration boxes
# It is a list of 4 1D arrays containing calibration box information for the 4 chips
#             chip sequence: (top left, top right, bottom left, bottom right)
#             each 1D array: (y start, y stop, x start, x stop)
back_calib_boxes = [[10, 110, 10, 110], [10, 110, 410, 510], [410, 510, 10, 110], [410, 510, 410, 510]]

# Fix seed for random number generation
np.random.seed(129)


# ==========================
# STEP-1: DATA PREPROCESSING
# ==========================

# Create a temporary folder to store intermediate data
temp_folder = 'temp_folder'
os.makedirs(temp_folder, exist_ok=True)

# Process data from one angle at a time and store
open_beam = None  # Initialized with None for the first angle, will be replaced by the actual open beam for other angles
all_angles_paths = h_preproc.generate_paths(proj_folder_path)
for i, angle in enumerate(angles):
    processed_data, open_beam = h_preproc.hyper_data_preprocessing(all_angles_paths[i],
                                                                   ob_folder_path=ob_folder_path,
                                                                   open_beam=open_beam,
                                                                   wave_idx_start=wave_idx_start,
                                                                   num_total_wave=num_total_wave,
                                                                   back_calib_boxes=back_calib_boxes)
    np.save(os.path.join(temp_folder, 'processed_data_' + str(angle) + '.npy'), processed_data)


# ==============================
# STEP-2: LARGE DATA DEHYDRATION
# ==============================

# Perform initial dehydration to estimate subspace basis vectors for each angle
subspace_basis_all_angles = []
for angle in angles:
    processed_data = np.load(os.path.join(temp_folder, 'processed_data_' + str(angle) + '.npy'))
    _, subspace_basis, _ = mj.hsnt.dehydrate(processed_data, num_materials=num_materials, verbose=verbose)
    subspace_basis_all_angles.append(subspace_basis)
subspace_basis_all_angles = np.concatenate(subspace_basis_all_angles, axis=0)

# Estimate refined set of subspace basis vectors combining estimations for all angles
_, subspace_basis, dataset_type = mj.hsnt.dehydrate(subspace_basis_all_angles, num_materials=num_materials, verbose=verbose)

# Perform final dehydration for each angle using the refined subspace basis vectors
subspace_data_all_angles = []
for angle in angles:
    processed_data = np.load(os.path.join(temp_folder, 'processed_data_' + str(angle) + '.npy'))
    subspace_data, _, _ = mj.hsnt.dehydrate(processed_data, subspace_basis=subspace_basis, verbose=verbose)
    subspace_data_all_angles.append(subspace_data)
subspace_data_all_angles = np.concatenate(subspace_data_all_angles, axis=0)

# Delete the temporary folder
shutil.rmtree(temp_folder)


# ===========================
# STEP-3: MBIR RECONSTRUCTION
# ===========================

# MBIR model setup
angles_r = np.array(angles) * np.pi / 180  # Convert the angles to radian
detector_rows, detector_columns, num_waves = open_beam.shape
mj_model = mj.ParallelBeamModel((len(angles), detector_rows, detector_columns), angles_r)
mj_model.set_params(snr_db=recon_snr_db, verbose=verbose)

# Perform MBIR
subspace_dimension = subspace_data_all_angles.shape[-1]
subspace_recons = []
for idx in range(subspace_dimension):
    print("Reconstructing data for subspace index: " + str(idx))
    subspace_recon, _ = mj_model.recon(subspace_data_all_angles[:, :, :, idx])
    subspace_recons.append(subspace_recon)
subspace_recons = np.moveaxis(np.array(subspace_recons), 0, -1)

# Pack dehydrated reconstructions
hsnt_dehydrated_recons = [subspace_recons, subspace_basis, dataset_type]

# Save data
metadata = mj.hsnt.create_hsnt_metadata(dataset_name=dataset_name)
mj.hsnt.export_hsnt_data_hdf5(output_file_name, hsnt_dehydrated_recons, metadata)


# ===========================================
# STEP-4: PARTIAL REHYDRATION & VISUALIZATION
# ===========================================

# Choose the middle wavelength bin and middle slice to view
disp_wave_idx = num_waves // 2
disp_slice = detector_rows // 2

# Rehydrate only the display wavelength reconstruction
hsnt_recon = mj.hsnt.rehydrate(hsnt_dehydrated_recons, hyperspectral_idx=disp_wave_idx)

# Plot image
plt.imshow(hsnt_recon[:, :, disp_slice, 0], cmap='gray', vmin=0, vmax=None)
plt.colorbar()
plt.show()
