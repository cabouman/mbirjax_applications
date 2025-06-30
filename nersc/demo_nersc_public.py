import sys
import subprocess
import importlib.util

# Install necessary packages
required_packages = ['pywavelets']
for package in required_packages:
    if importlib.util.find_spec(package) is None:
        print(f"{package} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}. Please install it manually.")
            sys.exit(1)
    else:
        print(f"{package} is already installed.")


import numpy as np
import jax.numpy as jnp
import pprint
import mbirjax as mj
import mbirjax.preprocess as mjp
import nersc_utils
import ring_utils
import h5py
import warnings

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('This script is a demonstration of the MBIR reconstruction workflow using public NERSC datasets.\n')

    # NERSC dataset Google Drive share link
    dataset_url = 'https://drive.google.com/file/d/1CpsiceN7zAjmeb07TKL4SbkW_5SHpgJS/view?usp=drive_link'
    # Destination path to download the NERSC data
    download_dir = './demo_data/'
    # Path to NERSC data directory
    dataset_dir = mj.download_and_extract(dataset_url, download_dir)

    # Load reconstruction parameters from data.
    with h5py.File(dataset_dir, "r") as data:
        num_det_rows = int(data['/measurement/instrument/detector/dimension_y'][0])
        num_det_channels = int(data['/measurement/instrument/detector/dimension_x'][0])
        num_views = int(data['/process/acquisition/rotation/num_angles'][0])
        pixel_size = data['/measurement/instrument/detector/pixel_size'][0] / 10.0
        angles = -np.deg2rad(data['exchange/theta'])
        obj_scan = data['exchange/data'][:]
        blank_scan = data['exchange/data_white'][:]
        dark_scan = data['exchange/data_dark'][:]

    # Set reconstruction parameters
    warnings.warn("No center of rotation provided. Using detector midpoint as default.")
    center_of_rotation = num_det_channels / 2
    sharpness = 1.0
    det_channel_offset = (center_of_rotation - num_det_channels / 2)

    # Select a subset of slices for reconstruction
    num_slices = 3
    mid_slice = num_det_rows // 2
    sino_used = (mid_slice - num_slices // 2, mid_slice + num_slices // 2)
    # Define sino_used = (0, num_slices) for full recon

    # Get (subsetted) object, blank, and dark scans
    obj_scan, blank_scan, dark_scan, _ = mjp.crop_view_data(
        obj_scan, blank_scan, dark_scan,
        crop_pixels_sides=0,
        crop_pixels_top=sino_used[0],
        crop_pixels_bottom=obj_scan.shape[1] - sino_used[1],
        defective_pixel_array=()
    )

    print("\n*******************************************************",
          "\n************** NERSC dataset preprocessing **************",
          "\n*******************************************************")
    # Compute sinogram data
    sinogram = mjp.compute_sino_transmission(obj_scan, blank_scan, dark_scan)

    # Remove stripe artifacts from sinogram
    sinogram = jnp.array(sinogram)
    sinogram = ring_utils.remove_stripe(sinogram)
    sinogram = ring_utils.remove_stripe_wavelet_fourier(sinogram)

    # Display the sinogram
    mj.slice_viewer(sinogram.transpose((0, 2, 1)), title='Original sinogram')

    print("\n*******************************************************",
          "\n***************** Set up MBIRJAX model ****************",
          "\n*******************************************************")
    # ParallelBeamModel constructor
    parallel_model = mj.ParallelBeamModel(sinogram_shape=sinogram.shape, angles=angles)

    # Set reconstruction parameter values
    parallel_model.set_params(sharpness=sharpness, det_channel_offset=det_channel_offset, verbose=1)

    # Padding the reconstruction size
    recon_shape = parallel_model.get_params("recon_shape")
    recon_row_scale = 1.3
    recon_col_scale = 1.3
    parallel_model.scale_recon_shape(row_scale=recon_row_scale, col_scale=recon_col_scale)

    # Print out model parameters
    parallel_model.print_params()

    print("\n*******************************************************",
          "\n************* Perform MBIR Reconstruction *************",
          "\n*******************************************************")

    recon, recon_dict = parallel_model.recon(sinogram)

    # Convert reconstruction values to units of 1/cm
    recon /= pixel_size

    # Undo padding of reconstruction
    center_x, center_y = recon.shape[0] // 2, recon.shape[1] // 2
    crop_size = num_det_channels // 2
    recon = recon[center_y - crop_size:center_y + crop_size, center_x - crop_size:center_x + crop_size]

    # Masking the reconstruction to display the circular ROR region
    circular_mask = nersc_utils.create_circular_mask(recon.shape[0], recon.shape[1]).astype(int)
    recon = recon * circular_mask[:, :, np.newaxis]

    # Save reconstruction results
    output_path = f'./demo_data/output/mbir_recon.h5'
    parallel_model.save_recon_hdf5(filepath=output_path, recon=recon, recon_dict=recon_dict)

    # Display the results
    mj.slice_viewer(recon, data_dicts=recon_dict, vmin = 0, vmax = 10, title=f'MBIR Reconstruction')