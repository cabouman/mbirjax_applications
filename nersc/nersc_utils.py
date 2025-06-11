import os, sys
import numpy as np
import re
import shutil
import tarfile
import gdown
import urllib.request
import urllib.error
from urllib.parse import urlparse

def download_and_extract(download_url, save_dir):
    """
    Download or copy a file from a URL or local file path. If it's a .tar file, extract it to the specified directory.
    Supports Google Drive links, regular HTTP/HTTPS URLs, and local file paths.
    If the file already exists in the save directory, the user will be prompted to decide whether to overwrite it.

    Parameters
    ----------
    download_url : str
        URL or local file path to the file. Supports:
        - Google Drive shared links
        - HTTP/HTTPS URLs
        - Local file paths
        If a URL, it must be public and accessible.
    save_dir : str
        Path to the directory where the file will be saved/copied and extracted (if tar).

    Returns
    -------
    result_path : str
        - For tar files: The path to the extracted top-level directory
        - For other files: The path to the downloaded/copied file

    Example
    -------
    >>> # Tar file extraction
    >>> extracted_dir = download_and_extract("https://example.com/data.tar.gz", "./data")
    >>> print(f"Extracted data is in: {extracted_dir}")

    >>> # Google Drive file download
    >>> file_path = download_and_extract("https://drive.google.com/file/d/1ABC123/view", "./data")
    >>> print(f"Downloaded file is at: {file_path}")

    >>> # Local file copy
    >>> result = download_and_extract("/path/to/local/data.tar.gz", "./data")
    >>> print(f"Result is in: {result}")
    """

    def is_google_drive_url(url):
        """Check if URL is a Google Drive link"""
        return "drive.google.com" in url

    def is_tar_file(filename):
        """Check if file is a tar archive based on extension"""
        tar_extensions = ['.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz']
        return any(filename.lower().endswith(ext) for ext in tar_extensions)

    def extract_google_drive_id(url):
        """Extract Google Drive file ID from URL"""
        pattern = r"(?:https?:\/\/)?(?:www\.)?drive\.google\.com\/(?:file\/d\/|open\?id=)([a-zA-Z0-9_-]+)"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        else:
            raise ValueError("Invalid Google Drive URL format")

    is_download = True
    parsed = urlparse(download_url)
    is_url = parsed.scheme in ('http', 'https')
    is_google_drive = is_url and is_google_drive_url(download_url)

    if is_google_drive:
        file_id = extract_google_drive_id(download_url)
        marker_file = os.path.join(save_dir, f".gdrive_{file_id}")

        if os.path.exists(marker_file):
            is_download = query_yes_no(
                f"\nGoogle Drive file (ID: {file_id}) has been downloaded before.\nDo you want to download it again?")

        filename = f"gdrive_{file_id}"
    else:
        filename = os.path.basename(parsed.path if is_url else download_url)

    if not is_google_drive:
        file_path = os.path.join(save_dir, filename)
        if os.path.exists(file_path):
            is_download = query_yes_no(
                f"\nFile named {file_path} already exists.\nDo you still want to download/copy and overwrite the file?")

    if is_download:
        os.makedirs(save_dir, exist_ok=True)

        if is_url:
            if is_google_drive:
                print("Downloading file from Google Drive...")
                try:
                    gdrive_url = f"https://drive.google.com/uc?id={file_id}"

                    downloaded_path = gdown.download(gdrive_url, output=None, quiet=False)
                    if downloaded_path and os.path.isfile(downloaded_path):
                        actual_filename = os.path.basename(downloaded_path)
                        target_path = os.path.join(save_dir, actual_filename)
                        shutil.move(downloaded_path, target_path)
                        file_path = target_path
                        filename = actual_filename

                        with open(marker_file, 'w') as f:
                            f.write(actual_filename)
                    else:
                        raise RuntimeError("Google Drive download failed or returned invalid path")

                    print(f"Download successful! File saved to {file_path}")
                except Exception as e:
                    raise RuntimeError(f"Google Drive download failed: {str(e)}")
            else:
                print("Downloading file...")
                try:
                    urllib.request.urlretrieve(download_url, file_path)
                except urllib.error.HTTPError as e:
                    if e.code == 401:
                        raise RuntimeError(f'HTTP {e.code}: authentication failed!')
                    elif e.code == 403:
                        raise RuntimeError(f'HTTP {e.code}: URL forbidden!')
                    elif e.code == 404:
                        raise RuntimeError(f'HTTP {e.code}: URL not found!')
                    else:
                        raise RuntimeError(f'HTTP {e.code}: {e.reason}')
                except urllib.error.URLError as e:
                    raise RuntimeError('URLError raised! Check internet connection.')
                print(f"Download successful! File saved to {file_path}")
        else:
            print(f"Copying local file from {download_url} to {file_path}...")
            if not os.path.isfile(download_url):
                raise RuntimeError(f"Provided file path does not exist: {download_url}")
            shutil.copy2(download_url, file_path)
            print(f"Copy successful! File saved to {file_path}")

        if is_tar_file(filename):
            print(f"Extracting tarball file to {save_dir}...")
            try:
                with tarfile.open(file_path, 'r') as tar_file:
                    tar_file.extractall(save_dir)
                print(f"Extraction successful!")

                top_level_dir = get_top_level_tar_dir(file_path)
                extracted_path = os.path.join(save_dir, top_level_dir)
                return extracted_path
            except Exception as e:
                raise RuntimeError(f"Failed to extract tar file: {str(e)}")
        else:
            return file_path

    if is_google_drive and not is_download:
        try:
            with open(marker_file, 'r') as f:
                actual_filename = f.read().strip()
            file_path = os.path.join(save_dir, actual_filename)
            filename = actual_filename
        except:
            file_path = os.path.join(save_dir, filename)

    if is_tar_file(filename):
        top_level_dir = get_top_level_tar_dir(file_path)
        file_path = os.path.join(save_dir, top_level_dir)

    return file_path

def get_top_level_tar_dir(tar_path, max_entries=10):
    """
    Determine the top-level directory inside a tarball file by sampling up to max_entries members.

    Parameters
    ----------
    tar_path : str
        Path to the tarball file.
    max_entries : int
        Maximum number of entries to sample.

    Returns
    -------
    dir_name : str
        The name of the top-level directory.
    """
    top_levels = set()

    with tarfile.open(tar_path, 'r') as tar:
        for i, member in enumerate(tar):
            if not member.name.strip():
                continue
            top_dir = member.name.split('/')[0]
            top_levels.add(top_dir)

            if len(top_levels) > 1 or i + 1 >= max_entries:
                break
    if len(top_levels) == 1:
        dir_name = top_levels.pop()
    else:
        raise ValueError("No top level directory found in {}".format(tar_path))
    return dir_name

def query_yes_no(question, default="n"):
    """
    Ask a yes/no question via input() and return the answer.

    Parameters
    ----------
    question : str
        The question presented to the user.
    default : str
        The default answer if the user just presses Enter ("y" or "n").

    Returns
    -------
    bool
        True for "yes" or Enter, False for "no".
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
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")
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
    if center is None:
        center = (int(width/2), int(height/2))
    if radius is None:
        radius = min(center[0], center[1], width-center[0], height-center[1])

    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask