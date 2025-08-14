import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import scipy
import mbirjax as mj
import mbirjax.preprocess as mjp

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('This script is a demonstration of the mbirjax metal artifact reduction (MAR) capability.\n')

    # Specify the directory containing the .nsipro file
    dataset_dir = "./demo_data/Autoinjector_HighRes_Horizontal"

    # Output path
    output_path = './output/lilly/'   # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

    filename = os.path.basename(dataset_dir)
    print(os.listdir(dataset_dir))

    # #### preprocessing parameters
    downsample_rate = [4, 4]  # downsample factor of scan images along detector rows and detector columns.
    subsample_view_factor = 4 # view subsample factor.

    # #### recon parameters
    sharpness = 1.0
    alpha = [1.0, 0.0, 0.0]  # beam_hardening_correction coefficient
    order = (3, 4)
    verbose = 1

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

    print("\n*************** Compute MAR reconstruction ***************")
    # Compute MAR reconstructions and plastic/metal segmentations
    recon = mjp.recon_BH_plastic_metal(ct_model, sino, weights_trans, order=order, verbose=0)
    plastic_mask, metal_mask, plastic_scale, metal_scale = mjp.segment_plastic_metal(recon)

    # print("\n*********** view plastic and metal masks *************")
    # mj.slice_viewer(plastic_mask, metal_mask, vmin=0, vmax=1.0, slice_axis=0, slice_label=['Plastic Mask', 'Metal Mask'], title="Final Plastic and Metal Masks")

    # Save recon to hdf5
    print("\n*********** save mar recon in h5 format *************")
    mj.export_recon_hdf5(os.path.join(output_path, f"recon_{filename}_mar.h5"), recon, recon_dict=None)

    # print("\n*********** view original and corrected reconstruction *************")
    # vmin = 0
    # vmax = downsample_rate[0] * 0.025
    # mj.slice_viewer(recon, vmin=0, vmax=vmax, slice_axis=0, slice_label=['MBIR MAR'], title='MAR reconstruction')
