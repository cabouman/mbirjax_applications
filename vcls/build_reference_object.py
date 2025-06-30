seed = 42  # Change this value to control randomness across runs

import numpy as np
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp
import os

"""
Restrict to center slices to investigate noise in center slices.  
"""

if __name__ == '__main__':
    #######################
    # Sets pointers to data
    #######################
    # path or URL to CT scan in h5 format with tgz wrapper
    dataset_url_scan = '/depot/bouman/data/ORNL/hfn_scan.tgz'
    # path to directory for storage of data
    download_dir = './demo_data/'

    ######################
    # Set recon parameters
    ######################
    sharpness = 1.0
    snr_db = 35.0
    max_iterations = 20


    ###############
    # Download Data
    ###############
    # Download and extract data
    dataset_dir_scan = mj.download_and_extract(dataset_url_scan, download_dir)

    #################
    # Construct model
    #################
    # Load and preprocess ORNL data
    # List all files ending in .h5 or .hdf5
    hdf5_files = sorted(
        f for f in os.listdir(dataset_dir_scan)
        if f.lower().endswith(('.h5', '.hdf5'))
    )
    filename = os.path.join(dataset_dir_scan, hdf5_files[0])
    full_sino, cone_beam_params, optional_params = mjp.pymbir.compute_sino_and_params(filename)

    # Construct cone beam object using ORNL parameters
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    # Set optional geometry parameters
    ct_model.set_params(**optional_params)
    # Set reconstruction parameters
    ct_model.set_params(sharpness=sharpness, snr_db=snr_db)

    # Do a recon with the full sinogram
    recon, recon_dict = ct_model.recon(full_sino, max_iterations=max_iterations)
    # mj.slice_viewer(recon, slice_axis=2)

    # Zero out distorted marginal slices at both ends
    indices = jnp.array(list(range(100)) + list(range(-100, 0)))
    recon = recon.at[:, :, indices].set(0)

    # Segment the reconstruction to obtain the reference object
    thresholds = mj.preprocess.multi_threshold_otsu(recon, classes=3)
    segmentation = np.digitize(recon, bins=thresholds)
    reference_object = (segmentation >= 2).astype(np.float32)

    # Store the reference object
    npy_dir = './demo_data/hfn_reference_object'
    os.makedirs(npy_dir, exist_ok=True)
    with open(os.path.join(npy_dir, f'hfn_reference_object.npy'), 'wb') as f:
        np.save(f, reference_object)

