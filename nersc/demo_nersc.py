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
import h5py

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('This script is a demonstration of the MBIR reconstruction workflow.\n')

    # Define available datasets and parameters
    available_datasets = {
        'public': {
            'url': 'https://drive.google.com/file/d/1CpsiceN7zAjmeb07TKL4SbkW_5SHpgJS/view?usp=drive_link',
            'det_channel_offset': 0.0,
            'ROR_scale': 1.2,
            'slice_number': None,
            'apply_mask': False,
        },
        'fuelcell': {
            'url': '/depot/bouman/data/nersc/demo_nersc_fuelcell.tgz',
            'det_channel_offset': 3.0,
            'ROR_scale': 1.2,
            'slice_number': 154,
            'apply_mask': True,
        },
        'permafrost': {
            'url': '/depot/bouman/data/nersc/demo_nersc_permafrost.tgz',
            'det_channel_offset': -71.125,
            'ROR_scale': 1.2,
            'slice_number': None,
            'apply_mask': True,
        },
        'HPcell': {
            'url': '/depot/bouman/data/nersc/nersc_HPcell_data.tgz',
            'det_channel_offset': 0.0,
            'ROR_scale': 1.2,
            'slice_number': None,
            'apply_mask': True,
        },

    }

    # Prompt user for dataset selection using a numbered menu
    dataset_names = list(available_datasets.keys())
    print("Available datasets:")
    for i, name in enumerate(dataset_names, 1):
        print(f"{i}. {name}")
    try:
        selection = int(input("Enter the number of the dataset to reconstruct: "))
        if 1 <= selection <= len(dataset_names):
            dataset = dataset_names[selection - 1]
        else:
            raise ValueError
    except ValueError:
        print("Invalid selection.")
        sys.exit(1)

    # Set values of data set specific parameters
    dataset_url = available_datasets[dataset]['url']
    det_channel_offset = available_datasets[dataset]['det_channel_offset']
    ROR_scale = available_datasets[dataset]['ROR_scale']
    slice_number = available_datasets[dataset]['slice_number']

    # Set reconstruction parameters
    sharpness = 1.0
    num_slices = 4

    # Download data
    download_dir = './demo_data/'
    dataset_dir = mj.download_and_extract(dataset_url, download_dir)

    # Load reconstruction parameters from data.
    with h5py.File(dataset_dir, "r") as data:
        # Get pixel size in units of cm
        pixel_size = data['/measurement/instrument/detector/pixel_size'][0] / 10.0
        angles = -np.deg2rad(data['exchange/theta'])
        obj_scan = data['exchange/data'][:]
        blank_scan = data['exchange/data_white'][:]
        dark_scan = data['exchange/data_dark'][:]

    # Determine number of detector rows to crop from top and bottom
    num_views, num_det_rows, num_det_channels = obj_scan.shape
    num_slices = np.minimum(num_det_rows, num_slices)
    if slice_number is None:
        crop_pixels_top = (num_det_rows - num_slices) // 2
        crop_pixels_bottom = (num_det_rows - num_slices) // 2
    else:
        crop_pixels_top = slice_number - num_slices // 2
        crop_pixels_bottom = num_det_rows - (slice_number + num_slices // 2)

    print("\n********** Crop out desired region of views **************")
    obj_scan, blank_scan, dark_scan, _ = mjp.crop_view_data(
        obj_scan, blank_scan, dark_scan,
        crop_pixels_sides=0, crop_pixels_top=crop_pixels_top, crop_pixels_bottom=crop_pixels_bottom,
        defective_pixel_array=()
    )

    print("\n********** Compute sinogram **************")
    sino = mjp.compute_sino_transmission(obj_scan, blank_scan, dark_scan)

    print("\n********** Remove stripe artifacts **************")
    sino = jnp.array(sino)
    sino = mjp.remove_all_stripe(sino)
    sino = mjp.remove_stripe_fw(sino)

    # Display the sinogram
    mj.slice_viewer(sino, slice_axis=1, title='Original sinogram')

    print("\n********** Construct parallel beam model **************")
    # ParallelBeamModel constructor
    ct_model = mj.ParallelBeamModel(sinogram_shape=sino.shape, angles=angles)
    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, det_channel_offset=det_channel_offset, verbose=1)

    # Scale the region of reconstruction (ROR) to reduce artifacts
    pad_size = ct_model.scale_recon_shape(row_scale=ROR_scale, col_scale=ROR_scale)
    print(f"Padding applied to rows, cols, and slices: {pad_size}")

    # Print out model parameters
    ct_model.print_params()

    print("\n********** Perform MBIR reconstruction **************")
    recon, recon_dict = ct_model.recon(sino)
    recon /= pixel_size # convert to units of 1/cm

    # Mask out Region of Interest (ROI)
    radial_margin = max(pad_size[0], pad_size[1])//2
    if available_datasets[dataset].get('apply_mask', False):
        recon = mjp.apply_cylindrical_mask(recon, radial_margin=radial_margin, top_margin=0, bottom_margin=0)

    # Save reconstruction results
    output_path = f'./demo_data/output/mbir_recon_{dataset}.h5'
    ct_model.save_recon_hdf5(filepath=output_path, recon=recon, recon_dict=recon_dict)

    # Display the results
    mj.slice_viewer(recon, data_dicts=recon_dict, vmin = 0, vmax = 10, title=f'MBIR Reconstruction - {dataset}')