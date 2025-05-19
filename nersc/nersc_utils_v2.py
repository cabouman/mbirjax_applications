import os, sys
import numpy as np
import re
import tarfile
import subprocess
import importlib.util

# List of required packages
required_packages = ['gdown', 'pywavelets']

def check_and_install_packages():
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

# Install necessary packages
check_and_install_packages()

def extract_tar(extract_url, save_dir):
    """Extract data file from "extract_url" and save it to "save_dir".
       If a corresponding ".h5" file exists in 'save_dir', it automatically detects and uses that file without extracting.

    Args:
        extract_url (str): The URL to extract data from.
        save_dir (str): The directory where the extracted file will be saved.

    Returns:
        h5_file_path (str): The os path to the data file in 'save_dir'.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Check if source file exists
    if not os.path.isfile(extract_url):
        raise FileNotFoundError(f"Source file not found: {extract_url}")

    # Extract the .tgz file
    print("Extracting file from provided URL...")
    with tarfile.open(extract_url) as tar:
        member = tar.getmembers()[0]
        h5_filename = member.name
        h5_file_path = os.path.join(save_dir, h5_filename)

        # Check if file already exists
        if os.path.exists(h5_file_path):
            print(f"File {h5_file_path} already exists, skipping.")
            return h5_file_path

        tar.extractall(path=save_dir)

    return h5_file_path

import gdown
def download_file(dataset_url, save_dir):
    """Downloads data file from "dataset_url" and saves it to `save_dir`.
       If a corresponding ".h5" file exists in 'save_dir', it automatically detects and uses that file without downloading.

    Args:
        dataset_url (str): The URL to download the data file from. The URL must be a Google Drive shared link.
            The URL must be public and accessible.
        save_dir (str): The directory where the downloaded file will be saved.

    Returns:
        h5_file_path (str): The os path to the data file in `save_dir`.
    """
    # Extract Google drive file ID
    pattern = r"(?:https?:\/\/)?(?:www\.)?drive\.google\.com\/(?:file\/d\/|open\?id=)([a-zA-Z0-9_-]+)"
    match = re.search(pattern, dataset_url)

    if match:
        file_id = match.group(1)
        dataset_url = f"https://drive.google.com/uc?id={file_id}"
    else:
        raise ValueError("Invalid Google Drive URL format")

    # Extract the file name from the URL
    h5_file_name = os.path.basename(dataset_url)
    h5_file_path = os.path.join(save_dir, h5_file_name)

    # Check if file already exists
    if os.path.exists(h5_file_path):
        print(f"h5 file already exists: {h5_file_path}")
        return h5_file_path

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Download the file
    print("Downloading file from provided URL...")
    gdown.download(dataset_url, h5_file_path, quiet=False)

    return h5_file_path


def create_circular_mask(height, width, center=None, radius=None):
    """ This function creates a 2D binary mask, which denotes the circular region specified by (height, width, center, radius).
    Args:
        height (int): height of the mask
        width (int): height of the mask
        center (tuple): [Default=None] central coordinates of the circle. If None, (int(width/2), int(height/2)) will be used as the center.
        radius (float): radius of the circle.
    Returns:
        3D boolean array denoting the circular region defined by center and radius. Any pixels inside the circular region will be marked with 1.
    """
    if center is None: # use the middle of the image
        center = (int(width/2), int(height/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], width-center[0], height-center[1])

    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask