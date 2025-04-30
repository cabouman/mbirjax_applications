import os, sys
import numpy as np
import urllib.request
import tarfile
import warnings
from urllib.parse import urlparse
import shutil

def download_and_extract_tar(download_url, save_dir):
    """
    Download or copy a .tar file from a URL or local file path, extract it to the specified directory, and return the path to the extracted top-level directory.

    If the tarball already exists in the save directory, the user will be prompted to decide whether to overwrite it.

    Parameters
    ----------
    download_url : str
        URL or local file path to the tarball. If a URL, it must be public.
    save_dir : str
        Path to the directory where the tarball will be saved/copied and extracted.

    Returns
    -------
    extracted_file_name : str
        The path to the extracted top-level directory.

    Example
    -------
    >>> extracted_dir = download_and_extract_tar("https://example.com/data.tar.gz", "./data")
    >>> print(f"Extracted data is in: {extracted_dir}")

    >>> extracted_dir = download_and_extract_tar("/path/to/local/data.tar.gz", "./data")
    >>> print(f"Extracted data is in: {extracted_dir}")
    """
    is_download = True
    parsed = urlparse(download_url)
    is_url = parsed.scheme in ('http', 'https')

    tarball_name = os.path.basename(parsed.path if is_url else download_url)
    tarball_path = os.path.join(save_dir, tarball_name)

    if os.path.exists(tarball_path):
        is_download = False  # query_yes_no(f"\nData named {tarball_path} already exists.\nDo you still want to download/copy and overwrite the file?")

    if is_download:
        os.makedirs(os.path.dirname(tarball_path), exist_ok=True)
        if is_url:
            print("Downloading file ...")
            try:
                urllib.request.urlretrieve(download_url, tarball_path)
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
            print(f"Download successful! Tarball file saved to {tarball_path}")
        else:
            print(f"Copying local file from {download_url} to {tarball_path} ...")
            if not os.path.isfile(download_url):
                raise RuntimeError(f"Provided file path does not exist: {download_url}")
            shutil.copy2(download_url, tarball_path)
            print(f"Copy successful! Tarball file saved to {tarball_path}")

        print(f"Extracting tarball file to {save_dir} ...")
        with tarfile.open(tarball_path, 'r') as tar_file:
            tar_file.extractall(save_dir)
        print(f"Extraction successful!")

    top_level_dir = get_top_level_tar_dir(tarball_path)
    extracted_file_name = os.path.join(save_dir, top_level_dir)

    return extracted_file_name

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
