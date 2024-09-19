import os, sys
import numpy as np
import urllib.request
import tarfile
import warnings

def download_and_extract_tar(download_url, save_dir):
    """Downloads a tarball file from `download_url`, extracts it to `save_dir`, and returns the file paths. 
       If an `.h5` file exists in `save_dir`, it automatically detects and uses that file without downloading or extracting.

    Args:
        download_url (str): The URL to download the tarball from. The URL must be public and accessible.
        save_dir (str): The directory where the downloaded file will be saved and extracted.

    Returns:
        tuple: A tuple containing:
            - tarball_path (str): The path to the tarball file saved in `save_dir`.
            - extracted_file_name (str): The name of the extracted top-level file or directory, or the detected `.h5` file.
    """

    # Local function to handle tarball extraction
    def extract_tarball(tarball_path, save_dir):
        print(f"Extracting tarball file to {save_dir} ...")
        try:
            with tarfile.open(tarball_path) as tar_file:
                extracted_file_name = os.path.join(save_dir, os.path.commonprefix(tar_file.getnames()))
                tar_file.extractall(save_dir)
                print(f"Extraction successful! File extracted to {extracted_file_name}")
            return extracted_file_name
        except Exception:
            warnings.warn(f"Extraction failed. Please make sure {tarball_path} is a tarball file.")
            return None

    # Check if .h5 file exists in save_dir
    if os.path.exists(save_dir):
        h5_files = sorted([f for f in os.listdir(save_dir) if f.endswith('.h5')])
        if h5_files:
            print(f"h5 file {h5_files[0]} detected, using that.")
            return None, os.path.join(save_dir, h5_files[0])

    # Prepare for download and extraction
    tarball_name = download_url.split('/')[-1]
    tarball_path = os.path.join(save_dir, tarball_name)

    # Check if tarball exists
    yes_download = not os.path.exists(tarball_path) or query_yes_no(f"{tarball_path} already exists. Do you want to overwrite?")

    # Download the tarball if needed
    if yes_download:
        os.makedirs(os.path.dirname(tarball_path), exist_ok=True)
        print("Downloading file ...")
        try:
            urllib.request.urlretrieve(download_url, tarball_path)
            print(f"Download successful! Tarball file saved to {tarball_path}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP status code {e.code}")
        except urllib.error.URLError:
            raise RuntimeError('URLError raised! Please check your internet connection.')

    # Extract tarball
    extracted_file_name = extract_tarball(tarball_path, save_dir)

    return tarball_path, extracted_file_name


# Temporarily move a copy of this here
def query_yes_no(question, default="n"):
    """Ask a yes/no question via input() and return the answer.
        Code modified from reference: `https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input/3041990`

    Args:
        question (string): Question that is presented to the user.
    Returns:
        Boolean value: True for "yes" or "Enter", or False for "no".
    """

    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = f" [y/n, default={default}] "
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
    return


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
