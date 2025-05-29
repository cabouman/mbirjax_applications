import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import scipy
import mbirjax as mj
import mbirjax.preprocess as mjp
import mar_utils

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('This script is a demonstration of the mbirjax metal artifact reduction (MAR) capability.\n')

    output_path = './output/nsi_demo_mar/'   # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

    # NSI file path
    dataset_url = '/depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal.tgz'
    # destination path to download and extract the NSI data and metadata.
    download_dir = './demo_data/'
    # Path to NSI scan directory.
    dataset_dir = mj.download_and_extract_tar(dataset_url, download_dir)

    # #### preprocessing parameters
    downsample_rate = [4, 4]  # downsample factor of scan images along detector rows and detector columns.
    subsample_view_factor = 8  # view subsample factor.

    # #### recon parameters
    sharpness = 1.0
    alpha = [1.0, 0.0, 0.0]  # beam_hardening_correction coefficient

    print("\n************** NSI dataset preprocessing **************")
    sino, cone_beam_params, optional_params = \
        mjp.nsi.compute_sino_and_params(dataset_dir, downsample_factor=downsample_rate, subsample_view_factor=subsample_view_factor)

    # #### beam hardening correction
    sino = mjp.BH_correction(sino, alpha=alpha)
    sino = jnp.maximum(sino, 0.0)   # Clip sinogram to be non-negative

    print("\n***************** Set up MBIRJAX model ****************")
    # ConeBeamModel constructor
    ct_model = mj.ConeBeamModel(**cone_beam_params)

    # Set additional geometry arguments
    ct_model.set_params(**optional_params)

    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, verbose=1, positivity_flag=True)
    weights_trans = ct_model.gen_weights(sino, weight_type='transmission_root')

    # Print out model parameters
    ct_model.print_params()

    recon = mar_utils.recon_BH_plastic_metal(ct_model, sino, weights_trans)
    recon_orig = ct_model.direct_recon(sino)

    print("\n*********** view original and corrected reconstruction *************")
    vmin = 0
    vmax = downsample_rate[0] * 0.025
    mj.slice_viewer(recon_orig, recon, vmin=0, vmax=vmax, slice_axis=0, slice_label=['FDK', 'MBIR MAR'], title='Comparison between the original and corrected reconstruction')
