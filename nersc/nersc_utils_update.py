import os, sys
import numpy as np
import gdown


def download_file(dataset_url, save_dir):
    """Downloads a fuelcell or permafrost file from "dataset_url" and saves it to `save_dir`.
       If a corresponding ".h5" file exists in 'save_dir', it automatically detects and uses that file without downloading.

    Args:
        dataset_url (str): The URL to download the fuelcell or permafrost file from. The URL must be a Google Drive shared link.
            The URL must be public and accessible.
        save_dir (str): The directory where the downloaded file will be saved.

    Returns:
        h5_file_path (str): The path to the fuelcell or permafrost file in `save_dir`.
    """
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