"""
Hyperspectral Neutron Tomography
--------------------------------

This script demonstrates hyperspectral neutron data preprocessing for Ni single-view datasets
"""

import os
import numpy as np
import mbirjax as mj
import preprocessing as h_preproc
import matplotlib.pyplot as plt

# Ni dataset information
proton_charge = '0_8c'  # Options: '0_8c', '1_6c', '2_4c', '4_8c', and '9_6c'
sample_type = 'cylinder'  # Options: 'cylinder' and 'flat'

# Setup paths
base_path = os.path.join('/depot/bouman/data/ORNL/hsnt/Ni_single_view', proton_charge)
ob_folder_path = os.path.join(base_path, 'open_beam')  # Raw open-beam folder path, may contain one or more observations
proj_folder_path = os.path.join(base_path, 'Ni_' + sample_type + '_projections')  # Raw projection folder path, may contain one or more views
output_file_name = ('processed_data_' + proton_charge + '_Ni_' + sample_type + '.h5')  # Output folder name

# Setup parameters
wave_idx_start = 100  # Index of the 1st wavelength bin to be loaded
num_total_wave = 200  # Number of total wavelength bins to be loaded

# Setup background calibration boxes
# It is a list of 4 1D arrays containing calibration box information for the 4 chips
#             chip sequence: (top left, top right, bottom left, bottom right)
#             each 1D array: (y start, y stop, x start, x stop)
back_calib_boxes = [[10, 110, 10, 110], [10, 110, 410, 510], [410, 510, 10, 110], [410, 510, 410, 510]]

# Fix seed for random number generation
np.random.seed(129)

# Preprocess the projection data (normalization and background offset correction)
processed_data = h_preproc.hyper_data_preprocessing(ob_folder_path,
                                                    proj_folder_path,
                                                    wave_idx_start=wave_idx_start,
                                                    num_total_wave=num_total_wave,
                                                    back_calib_boxes=back_calib_boxes)

# Save data
metadata = mj.hsnt.create_hsnt_metadata(dataset_name=proton_charge + '_Ni_' + sample_type + '_dataset',
                                        dataset_type="attenuation")
mj.hsnt.export_hsnt_data_hdf5(output_file_name, processed_data, metadata)

# Sample processed image
plt.imshow(processed_data[0, :, :, (num_total_wave - wave_idx_start) // 2], cmap='gray', vmin=0, vmax=None)
plt.colorbar()
plt.show()
