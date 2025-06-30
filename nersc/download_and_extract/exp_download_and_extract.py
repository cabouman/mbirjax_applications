import os
import sys
import subprocess
import importlib.util
import h5py
import mbirjax as mj


def main():
    # Install gdown package
    if importlib.util.find_spec('gdown') is None:
        print(f"gdown not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", 'gdown'])
            print(f"Successfully installed gdown.")
        except subprocess.CalledProcessError:
            print(f"Failed to install gdown. Please install it manually.")
            sys.exit(1)
    else:
        print(f"gdown is already installed.")


    ### Test tar file downloading and extraction from URL
    # file path
    dataset_url = 'https://engineering.purdue.edu/~bouman/data_repository/data/nersc-sand.tgz'
    # Destination path to download and extract the data
    download_dir = '../demo_data/exp_download_and_extract'

    print("Testing tar file downloading and extraction from URL...")

    try:
        # Path to data directory
        dataset_dir = mj.utilities.download_and_extract(dataset_url, download_dir)

        # Check if the extracted file exists
        if not os.path.exists(dataset_dir):
            print("Failed to extract file. File does not exist at expected location.")
        else:
            # Get parameters from data
            with h5py.File(dataset_dir, "r") as data:
                num_det_rows = int(data['/measurement/instrument/detector/dimension_y'][0])
                num_det_channels = int(data['/measurement/instrument/detector/dimension_x'][0])
                num_views = int(data['/process/acquisition/rotation/num_angles'][0])
                pixel_size = data['/measurement/instrument/detector/pixel_size'][0] / 10.0
                propagation_dist = data['/measurement/instrument/camera_motor_stack/setup/camera_distance'][1]
                kev = data['/measurement/instrument/monochromator/energy'][0] / 1000
                angular_range = data['/process/acquisition/rotation/range'][0]

            print("Successfully extracted and loaded parameters. File has been correctly processed!")
            print(
                f"{dataset_dir}: \
                        \n\t slices: {num_det_rows}, rays: {num_det_channels}, angles: {num_views}, angular range: {angular_range}°\
                        \n\t pixel size: {pixel_size * 10000:.3f} μm, distance: {propagation_dist:.3f} mm, energy: {kev:.2f} keV"
            )
            print()
            print("=" * 60)
            print()

    except Exception as e:
        print(f"Failed to extract or load parameters. Error: {e}")
        print()
        print("=" * 60)
        print()


    ### Test local tar file downloading and extraction
    # file path
    dataset_url = '/depot/bouman/data/nersc/demo_nersc_permafrost.tgz'
    # Destination path to download and extract the data
    download_dir = '../demo_data/exp_download_and_extract'

    print("Testing local tar file downloading and extraction...")

    try:
        # Path to data directory
        dataset_dir = mj.utilities.download_and_extract(dataset_url, download_dir)

        # Check if the extracted file exists
        if not os.path.exists(dataset_dir):
            print("Failed to extract file. File does not exist at expected location.")
        else:
            # Get parameters from data
            with h5py.File(dataset_dir, "r") as data:
                num_det_rows = int(data['/measurement/instrument/detector/dimension_y'][0])
                num_det_channels = int(data['/measurement/instrument/detector/dimension_x'][0])
                num_views = int(data['/process/acquisition/rotation/num_angles'][0])
                pixel_size = data['/measurement/instrument/detector/pixel_size'][0] / 10.0
                propagation_dist = data['/measurement/instrument/camera_motor_stack/setup/camera_distance'][1]
                kev = data['/measurement/instrument/monochromator/energy'][0] / 1000
                angular_range = data['/process/acquisition/rotation/range'][0]

            print("Successfully extracted and loaded parameters. File has been correctly processed!")
            print(
                f"{dataset_dir}: \
                        \n\t slices: {num_det_rows}, rays: {num_det_channels}, angles: {num_views}, angular range: {angular_range}°\
                        \n\t pixel size: {pixel_size * 10000:.3f} μm, distance: {propagation_dist:.3f} mm, energy: {kev:.2f} keV"
            )
            print()
            print("=" * 60)
            print()

    except Exception as e:
        print(f"Failed to extract or load parameters. Error: {e}")
        print()
        print("=" * 60)
        print()


    ### Test Google Drive file downloading and extraction
    # file path
    dataset_url = 'https://drive.google.com/file/d/1CpsiceN7zAjmeb07TKL4SbkW_5SHpgJS/view?usp=drive_link'
    # Destination path to download and extract the data
    download_dir = '../demo_data/exp_download_and_extract'

    print("Testing Google Drive file downloading and extraction...")

    try:
        # Path to data directory
        dataset_dir = mj.utilities.download_and_extract(dataset_url, download_dir)

        # Check if the extracted file exists
        if not os.path.exists(dataset_dir):
            print("Failed to extract file. File does not exist at expected location.")
        else:
            # Get parameters from data
            with h5py.File(dataset_dir, "r") as data:
                num_det_rows = int(data['/measurement/instrument/detector/dimension_y'][0])
                num_det_channels = int(data['/measurement/instrument/detector/dimension_x'][0])
                num_views = int(data['/process/acquisition/rotation/num_angles'][0])
                pixel_size = data['/measurement/instrument/detector/pixel_size'][0] / 10.0
                propagation_dist = data['/measurement/instrument/camera_motor_stack/setup/camera_distance'][1]
                kev = data['/measurement/instrument/monochromator/energy'][0] / 1000
                angular_range = data['/process/acquisition/rotation/range'][0]

            print("Successfully extracted and loaded parameters. File has been correctly processed!")
            print(
                f"{dataset_dir}: \
                        \n\t slices: {num_det_rows}, rays: {num_det_channels}, angles: {num_views}, angular range: {angular_range}°\
                        \n\t pixel size: {pixel_size * 10000:.3f} μm, distance: {propagation_dist:.3f} mm, energy: {kev:.2f} keV"
            )
            print()
            print("=" * 60)
            print()

    except Exception as e:
        print(f"Failed to extract or load parameters. Error: {e}")
        print()
        print("=" * 60)
        print()


if __name__ == '__main__':
    main()