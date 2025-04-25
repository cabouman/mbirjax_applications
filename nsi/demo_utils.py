import os, sys
import numpy as np
import urllib.request
import tarfile
import warnings

def download_and_extract_tar(download_url, save_dir):
    """ Given a download url, download the tarball file from ``download_url`` , extract the tarball to ``save_dir``, and return the paths to the tarball file as well as the extracted file. 
        If the file already exists in ``save_dir``, user will be queried whether it is desired to download and overwrite the existing files.
        ``download_url`` is assumed to have the format <url/{tarball_name}>.
        The tarball file is assumed to contain a single top-level directory.
 
    Args:
        download_url: An url to download the data. This url needs to be public.
        save_dir (string): Path to parent directory where downloaded file will be saved and extracted to. 
    Return:
        A tuple containing:
            - path to the tarball file. This will be ``save_dir``+ downloaded_file_name.
            - A list containing the names of the top level files 
    """

    is_download = True
    # the download url is assumed to have the format "**/{tarball_name}"
    tarball_name = download_url.split('/')[-1]
    # full path to the tarball file 
    tarball_path = os.path.join(save_dir, tarball_name)
    
    # If the tarball already exists, then prompt user whether to download and overwrite the existing file.
    if os.path.exists(tarball_path):
        is_download = query_yes_no(f"{tarball_path} already exists. Do you still want to download and overwrite the file?")
    
    ################### Download and extract tarball file
    if is_download:
        # make the directory where the tarball will be saved, if necessary.
        os.makedirs(os.path.dirname(tarball_path), exist_ok=True)
        ###### download the tarball
        print("Downloading file ...")
        try:
            urllib.request.urlretrieve(download_url, tarball_path)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise RuntimeError(
                    f'HTTP status code {e.code}: URL authentication failed! Currently we do not support downloading data from a url that requires authentication.')
            elif e.code == 403:
                raise RuntimeError(
                    f'HTTP status code {e.code}: URL forbidden! Please make sure the provided URL is public.')
            elif e.code == 404:
                raise RuntimeError(
                    f'HTTP status code {e.code}: URL not Found! Please check and make sure the download URL provided is correct.')
            else:
                raise RuntimeError(
                    f'HTTP status code {e.code}: {e.reason}. For more details please refer to https://en.wikipedia.org/wiki/List_of_HTTP_status_codes')
        except urllib.error.URLError as e:
            raise RuntimeError('URLError raised! Please check your internet connection.')
        
        # download is successful if no exceptions occur
        print(f"Download successful! Tarball file saved to {tarball_path}")

        ###### Extract to save_dir.
        print(f"Extracting tarball file to {save_dir} ...")
        with tarfile.open(tarball_path, 'r') as tar_file:
            tar_file.extractall(save_dir)
        print(f"Extraction successful!")

    # Get top level file names without extracting the tarball
    top_level_dir = get_top_level_tar_dir(tarball_path)
    extracted_file_name = os.path.join(save_dir, top_level_dir)
    
    return extracted_file_name


def get_top_level_tar_dir(tar_path, max_entries=10):
    """
    Determine the top level directory of the tarball file by getting up to max_entries files and finding
    a common prefix.

    Args:
        tar_path: Path to the tarball file.
        max_entries: Max number of entries to interrogate.

    Returns:
        The top level directory of the tarball file.
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

# # Example usage
# tar_path = 'your_archive.tar'
# top_level = get_top_level_dir_sampled(tar_path)
# if top_level:
#     print(f"Top-level directory (consistent across first 10): {top_level}")
# else:
#     print("Multiple top-level entries or insufficient data to determine.")


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
