# Hyperspectral Neutron Tomography Library - Preprocessing
# Copyright (C) 2021, Charles A Bouman.
# All rights reserved.

import os
import re
import glob
import warnings
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
import numpy as np
import tifffile
from scipy.signal import fftconvolve


def hyper_data_preprocessing(ob_folder_path, proj_folder_path, wave_idx_start=0, num_total_wave=None, ob_smoothing=True,
                             ob_smoothing_filter_width=3, back_calib_boxes=None):
    """Function to preprocess hyperspectral neutron data and produce open-beam normalized and background offset
    corrected projection densities.

    Args:
        ob_folder_path(str): base folder location of hyperspectral raw open-beam counts (tiff)
        proj_folder_path(str): base folder location of hyperspectral raw projection counts (tiff)
        wave_idx_start(int,optional): starting wavelength index
        num_total_wave(int,optional): number of wavelengths
        ob_smoothing(bool,optional): true/false, smooth open-beam if set to true
        ob_smoothing_filter_width(int,optional): width of the Hamming filter used for smoothing (must be an odd number)
        back_calib_boxes(list): list of 4 1D arrays containing calibration box information for the 4 chips
            chip sequence: (top left, top right, bottom left, bottom right)
            each 1D array: (y start, x start, y stop, x stop)

    Returns:
        ndarray: processed data with shape (num angles x height x width x wavelengths)
        """
    # Generate the input paths
    ob_paths = generate_path(ob_folder_path)
    proj_paths = generate_path(proj_folder_path)

    print('......Starting open-beam data processing......')

    ob_all_sets = []
    for i in range(len(ob_paths)):
        ob_all_sets.append(load_data(ob_paths[i], wave_idx_start=wave_idx_start, num_total_wave=num_total_wave))

    ob_all_sets = np.array(ob_all_sets).astype(np.float32)

    # Average over all the open-beam data sets
    open_beam = np.mean(ob_all_sets, axis=0)

    # Replace zeros in the open-beam data
    open_beam = replace_zero(open_beam)

    # Process open-beam data
    if ob_smoothing:
        open_beam = smooth_open_beam(open_beam, filter_width=ob_smoothing_filter_width)

    print('Open-beam data processing done......')

    print('......Starting projection data processing......')

    if back_calib_boxes is None:
        warnings.warn("Background offset correction skipped, required background calibration boxes not provided.")

    # We are going to process data one view at a time for multi-view data
    processed_data = []
    for idx in range(len(proj_paths)):
        print('Currently processing projection data from view index: ', idx)

        # Load raw projection data
        raw_projection = load_data(proj_paths[idx], wave_idx_start=wave_idx_start, num_total_wave=num_total_wave)

        # Replace zeros in the data
        raw_projection = replace_zero(raw_projection)

        # Normalize projection data
        norm_projection = normalize_projection(raw_projection, open_beam)

        # Perform background calibration
        if back_calib_boxes is not None:
            norm_projection = calibrate_background(norm_projection, back_calib_boxes)

        processed_data.append(norm_projection)

    processed_data = np.array(processed_data).astype(np.float32)

    print('Projection data processing done......')

    return processed_data


def generate_path(base_folder_path):
    """Function to generate folder paths for loading hyperspectral data. If there are no folders inside, it will return
    the base folder path only.

    Args:
        base_folder_path(str): base location where the folders for hyperspectral data are located

    Returns:
        list: contains all the generated folder paths or just the base folder path if no folders inside
        """
    # List and sort all the folders in the base directory
    folders = sorted([f for f in os.listdir(base_folder_path)
                      if os.path.isdir(os.path.join(base_folder_path, f))
                      and not f.startswith('.')], key=_natural_sort_key)

    # Return a list with the base folder path if no folders inside
    if len(folders) == 0:
        return [base_folder_path]

    # Return a list of all the folder paths otherwise
    return [os.path.join(base_folder_path, folder) for folder in folders]


def load_data(folder_path, wave_idx_start=0, num_total_wave=None):
    """Function to read tiff images from a range of wavelengths from the given location and return the data in a 3D array.

    Args:
        folder_path(str): folder location of the data to be loaded
        wave_idx_start(int,optional): starting wavelength index
        num_total_wave(int,optional): number of wavelengths

    Returns:
        ndarray: loaded data (height x width x wavelengths)
        """
    # Generate file paths
    file_paths = glob.glob(folder_path + '/*.tif')

    # Return if no tiff files available
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No TIFF files found in: {folder_path}")

    # Consider files from all the wavelengths if number of wavelengths is not provided
    if num_total_wave is None:
        num_total_wave = len(file_paths) - wave_idx_start

    # Active file paths
    active_file_paths = sorted(file_paths, key=_natural_sort_key)[wave_idx_start:wave_idx_start + num_total_wave]

    # Multiprocessing parameters
    num_process = min(max(1, cpu_count() - 1), len(active_file_paths))

    with Pool(num_process) as pool:
        count_data = pool.map(_load_tiff, active_file_paths)

    count_data = np.swapaxes(np.array(count_data), 0, 2)

    return count_data


def replace_zero(hyper_image):
    """Function to replace any zero value in the data array with the median of the closest 8 neighboring pixels at the
    same wavelength.

    Args:
        hyper_image(ndarray): 3D data array (height x width x wavelengths)

    Returns:
        ndarray: the 3D data array after replacing zeros (height x width x wavelengths)
        """
    corrected_image = np.copy(hyper_image)
    num_row, num_column, _ = corrected_image.shape

    # Declare epsilon
    epsilon = 1e-8
    
    # Find out the indices where the value is zero
    zero_idx = np.argwhere(corrected_image == 0)

    # Define the neighborhood
    row = zero_idx[:, 0]
    column = zero_idx[:, 1]
    wave_idx = zero_idx[:, 2]
    row_lb = row - 1  # Lower bound for row
    column_lb = column - 1  # Lower bound for column
    row_ub = row + 1  # Upper bound for row
    column_ub = column + 1  # Upper bound for column

    # For the out-of-bound pixel locations, use the nearest boundary pixel values (reflective boundary condition)
    row_lb[row_lb < 0] = 0
    column_lb[column_lb < 0] = 0
    row_ub[row_ub > (num_row - 1)] = num_row - 1
    column_ub[column_ub > (num_column - 1)] = num_column - 1

    # Replace zero values with the median of the closest 8 neighboring pixels
    corrected_image[row, column, wave_idx] = np.median((corrected_image[row_lb, column_lb, wave_idx],
                                                        corrected_image[row_lb, column, wave_idx],
                                                        corrected_image[row_lb, column_ub, wave_idx],
                                                        corrected_image[row, column_lb, wave_idx],
                                                        corrected_image[row, column_ub, wave_idx],
                                                        corrected_image[row_ub, column_lb, wave_idx],
                                                        corrected_image[row_ub, column, wave_idx],
                                                        corrected_image[row_ub, column_ub, wave_idx]), axis=0)

    # Find out the indices of the remaining zeros
    zero_idx = np.argwhere(corrected_image == 0)
    
    # Replace rest of the zeros with epsilon
    row = zero_idx[:, 0]
    column = zero_idx[:, 1]
    wave_idx = zero_idx[:, 2]
    corrected_image[row, column, wave_idx] = epsilon

    return corrected_image


def smooth_open_beam(open_beam, filter_width=5):
    """Function to smooth 3D open-beam data (height x width x wavelengths) by filtering with a 2D normalized hamming
    window for each wavelength.

    Args:
        open_beam(ndarray): 3D open-beam data (height x width x wavelengths)
        filter_width(int,optional): width of the Hamming filter used for smoothing (must be an odd number)

    Returns:
        ndarray: smooth 3D open-beam data (height x width x wavelengths)
        """
    if (filter_width % 2) == 0:
        filter_width += 1

    open_beam_smooth = np.empty_like(open_beam)

    hamming1d = np.hamming(filter_width)
    hamming2d = np.sqrt(np.outer(hamming1d, hamming1d))
    hamming2d = hamming2d / np.sum(hamming2d)

    pad = filter_width // 2

    for wave_idx in range(open_beam.shape[2]):
        img = open_beam[:, :, wave_idx]
        padded = np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
        smooth_padded = fftconvolve(padded, hamming2d, mode="same")
        open_beam_smooth[:, :, wave_idx] = smooth_padded[pad:pad + img.shape[0], pad:pad + img.shape[1]]

    return open_beam_smooth


def normalize_projection(raw_projection, open_beam):
    """Function to normalize 3D raw projection data (height x width x wavelengths) at all wavelengths by taking the
    negative log of the ratio of the raw projection and the corresponding open-beam.
    
    Args:
        raw_projection(ndarray): raw 3D projection data (height x width x wavelengths)
        open_beam(ndarray): processed 3D open-beam data (height x width x wavelengths)
    
    Returns:
        ndarray: normalized 3D projection densities (height x width x wavelengths)
        """
    # Taking the negative log of the ratio of raw projection and open-beam
    eps = 1e-8
    norm_projection = -np.log(np.maximum(raw_projection, eps) / np.maximum(open_beam, eps))

    # Replacing nan, +inf, and -inf with 0
    norm_projection = np.nan_to_num(norm_projection, nan=0, posinf=0, neginf=0)

    return norm_projection


def calibrate_background(norm_projection, back_calib_boxes):
    """Function to estimate background offsets caused by the mismatch of open-beam and raw projection counts at all
    wavelengths and remove the offsets from the data. Four boxed regions from the four chips where there are no objects
    are used to estimate the background offsets.
    
    Args:
        norm_projection(ndarray): normalized 3D projection densities (height x width x wavelengths)
        back_calib_boxes(list): list of 4 1D arrays containing calibration box information for the 4 chips
            chip sequence: (top left, top right, bottom left, bottom right)
            each 1D array: (y start, y stop, x start, x stop)

    Returns:
        ndarray: background offset corrected projection densities (height x width x wavelengths)
        """
    if len(back_calib_boxes) != 4:
        raise ValueError("back_calib_boxes must contain exactly 4 boxes.")

    # Initialize background calibrated projection
    back_calib_projection = np.empty_like(norm_projection, dtype=np.float32)

    # Define the chip starting/ending points
    height, width, _ = norm_projection.shape

    chips = [[0, height // 2, 0, width // 2],
             [0, height // 2, width // 2, width],
             [height // 2, height, 0, width // 2],
             [height // 2, height, width // 2, width]]

    # Compute and subtract background offset for each chip
    for i, back_calib_box in enumerate(back_calib_boxes):
        # Setup box
        b_y_start = back_calib_box[0]
        b_y_stop = back_calib_box[1]
        b_x_start = back_calib_box[2]
        b_x_stop = back_calib_box[3]
        box = norm_projection[b_y_start:b_y_stop, b_x_start:b_x_stop, :]

        # Compute background offset and subtract
        background_offset = np.mean(box, axis=(0, 1), keepdims=True).astype(np.float32)
        back_calib_projection[chips[i][0]:chips[i][1], chips[i][2]:chips[i][3]] = \
            (norm_projection[chips[i][0]:chips[i][1], chips[i][2]:chips[i][3]] - background_offset).astype(np.float32)

    return back_calib_projection


def _load_tiff(file_path):
    return tifffile.imread(file_path).astype(np.float32)


def _natural_sort_key(path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', os.path.basename(path))]
