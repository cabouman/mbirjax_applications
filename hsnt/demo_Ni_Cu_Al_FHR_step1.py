"""
Hyperspectral Neutron Tomography
--------------------------------

Step 1 for the Ni-Cu-Al FHR demo: preprocess raw projection data and write
per-angle processed arrays to the tmp folder.
"""

import os
import numpy as np
import hsnt_prep_utils as h_preproc


# Setup paths
base_path = '/depot/bouman/data/ORNL/hsnt/tci_2025_Ni_Cu_Al'
ob_folder_path = os.path.join(base_path, 'open_beam')  # Raw open-beam folder path, may contain one or more observations
proj_folder_path = os.path.join(base_path, 'projections')  # Raw projection folder path, may contain one or more views
tmp_folder = 'tmp'

# Setup parameters
wave_idx_start = 100  # Index of the 1st wavelength bin to be loaded
num_total_wave = 1200  # Number of total wavelength bins to be loaded
angles = [0.0, 6.2, 12.399, 16.231, 22.43, 32.461, 38.661, 42.492, 48.692, 58.723,
          64.922, 74.953, 81.153, 84.984, 91.184, 101.215, 107.415, 111.246, 117.446,
          127.477, 133.676, 143.707, 149.907, 153.738, 159.938, 169.969, 176.168]
roi = [50, -50, 50, -50]  # region of interest (y start, y stop, x start, x stop)
verbose = 0  # Print nothing if 0

# Setup background calibration boxes
# It is a list of 4 1D arrays containing calibration box information for the 4 chips
#             chip sequence: (top left, top right, bottom left, bottom right)
#             each 1D array: (y start, y stop, x start, x stop)
back_calib_boxes = [[10, 110, 10, 110], [10, 110, 410, 510], [410, 510, 10, 110], [410, 510, 410, 510]]

# Fix seed for random number generation
np.random.seed(129)


def main():
    print("--------------------------")
    print("STEP-1: DATA PREPROCESSING")
    print("--------------------------")

    os.makedirs(tmp_folder, exist_ok=True)

    # Process data from one angle at a time and store
    open_beam = None  # Initialized with None for the first angle, then reused for other angles
    all_angles_paths = h_preproc.generate_paths(proj_folder_path)
    for i, angle in enumerate(angles):
        print("Currently processing data from angle: ", angle)
        processed_data, open_beam = h_preproc.hyper_data_preprocessing(all_angles_paths[i],
                                                                       ob_folder_path=ob_folder_path,
                                                                       open_beam=open_beam,
                                                                       wave_idx_start=wave_idx_start,
                                                                       num_total_wave=num_total_wave,
                                                                       back_calib_boxes=back_calib_boxes,
                                                                       verbose=verbose)
        processed_data = processed_data[:, roi[0]: roi[1], roi[2]: roi[3]]
        np.save(os.path.join(tmp_folder, 'processed_data_' + str(angle) + '.npy'), processed_data)


if __name__ == '__main__':
    main()
