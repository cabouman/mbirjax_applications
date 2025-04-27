import numpy as np
import os
import time
import pprint
import jax
import jax.numpy as jnp
import jax.lax as lax


import mbirjax
import demo_utils
import pprint

pp = pprint.PrettyPrinter(indent=4)



def make_gaussian_kernel(sigma, size=None):
    if size is None:
        size = int(2 * 3 * sigma + 1)  # Cover approximately +/- 3 sigma
    coords = jnp.arange(size) - (size - 1) / 2
    gauss_1d = jnp.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss_1d /= gauss_1d.sum()
    kernel_2d = jnp.outer(gauss_1d, gauss_1d)
    kernel_2d /= jnp.sum(kernel_2d)  # Normalize to sum to 1.0
    return kernel_2d


def gaussian_blur(image, sigma):
    kernel = make_gaussian_kernel(sigma)
    kernel = kernel[:, :, None, None]  # Shape (H, W, in_channels=1, out_channels=1)
    image = image[None, :, :, None]    # Shape (batch=1, H, W, channels=1)
    blurred = lax.conv_general_dilated(
        image,
        kernel,
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    return blurred[0, :, :, 0]  # Remove batch and channel dims


def bhs_correction(sino, alpha, beta, sigma, batch_size=16, atten_factor=4):
    """
    Apply BHS correction to the input sinogram.

    Args:
        sino: jnp.ndarray of shape (views, rows, cols)
        alpha: float, beam hardening correction parameter
        beta: float, scatter correction parameter
        sigma: float, standard deviation for gaussian blur
        batch_size: int, number of views to process at a time
        atten_factor: float, factor to divide min attenuation, default is 4

    Returns:
        corrected_sino: jnp.ndarray of shape (views, rows, cols)
    """
    views, rows, cols = sino.shape

    # Step 1: Beam hardening correction
    sino = sino + alpha * jnp.power(sino, 2)

    # Step 2: Compute global min attenuation
    max_sino = jnp.max(sino)
    min_attenuation = jnp.exp(-max_sino) / atten_factor

    corrected = []

    for i in range(0, views, batch_size):
        sino_batch = sino[i:i+batch_size]

        # Step 3: Scatter correction
        attenuation = jnp.exp(-sino_batch)

        one_minus_attenuation = 1.0 - attenuation
        blurred = jax.vmap(lambda img: gaussian_blur(img, sigma))(one_minus_attenuation)

        scatter = beta * blurred

        corrected_attenuation = attenuation - scatter

        # Clip corrected attenuation to minimum attenuation value
        corrected_attenuation = jnp.maximum(corrected_attenuation, min_attenuation)

        corrected_batch = -jnp.log(corrected_attenuation)

        corrected.append(corrected_batch)

    corrected_sino = jnp.concatenate(corrected, axis=0)

    return corrected_sino


if __name__ == "__main__":
    print('This script is demonstrates the preprocessing and reconstruction of NSI an dataset\
    \n\t using both FDK and MBIR reconstruction.\n')

    # #### User defined params
    output_path = './output/nsi_demo/'  # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

    # path to store and extract the NSI data and metadata.
    download_dir = './demo_data/'

    # #### Prompt the user for dataset choice
    choice = input("Download dataset with metal? (Y/n): ").strip().lower()
    if choice == 'n':
        # URL to test phantom without metal
        dataset_url = 'https://www.datadepot.rcac.purdue.edu/bouman/data/demo_nsi_vert_no_metal_all_views.tgz'
        metal = False
    else:
        # URL to test phantom with metal
        dataset_url = 'https://www.datadepot.rcac.purdue.edu/bouman/data/demo_nsi_vert_metal_all_views.tgz'
        metal = True
    print(f"Selected dataset URL: {dataset_url}")

    # #### Download and extract data. Then set path to NSI scan directory.
    dataset_dir = demo_utils.download_and_extract_tar(dataset_url, download_dir)

    # #### preprocessing parameters
    downsample_factor = [4, 4]  # downsample factor of scan view images along detector rows and detector columns.
    subsample_view_factor = 8  # view subsample factor.

    # #### recon parameters
    sharpness = 5.0
    snr_db = 30.0
    bh_coefficient = 0.2  # beam_hardening_correction coefficient


    print("\n*******************************************************",
          "\n************** NSI dataset preprocessing **************",
          "\n*******************************************************")
    sino, cone_beam_params, optional_params = \
        mbirjax.preprocess.nsi.compute_sino_and_params(dataset_dir,
                                                       downsample_factor=downsample_factor,
                                                       subsample_view_factor=subsample_view_factor)
    # #### beam hardening correction
    if metal:
        sino = bhs_correction(sino, alpha=0.2, beta=0.00, sigma=2)

    print("\n*******************************************************",
          "\n***************** Set up MBIRJAX model ****************",
          "\n*******************************************************")
    # Construct cone beam object using NSI parameters
    ct_model = mbirjax.ConeBeamModel(**cone_beam_params)

    # Set optional NSI geometry parameters
    ct_model.set_params(**optional_params)

    # Set user determined parameter values
    ct_model.set_params(sharpness=sharpness, snr_db=snr_db, verbose=1)

    # Print out model parameters
    ct_model.print_params()

    print("\n*******************************************************",
          "\n************** Calculate sinogram weights *************",
          "\n*******************************************************")
    weights = ct_model.gen_weights(sino, weight_type='transmission_root')

    print("\n******************************************************",
          "\n************** Perform FDK reconstruction ************",
          "\n******************************************************")

    # ##########################
    # Perform FDK reconstruction
    fdk_recon = ct_model.direct_recon(sino)
    #mbirjax.slice_viewer(fdk_recon)

    print("\n*******************************************************",
          "\n************** Perform MBIR reconstruction ************",
          "\n*******************************************************")

    # #### Perform MBIR reconstruction
    time0 = time.time()
    mbir_recon, mbir_recon_params = ct_model.recon(sino, weights=weights)
    mbir_recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))

    # #### Print out parameters used in recon
    pprint.pprint(mbir_recon_params._asdict())

    # #### Save MBIR reconstruction to HDF5 file output
    mbirjax.preprocess.export_recon_to_hdf5(mbir_recon, os.path.join(output_path, "recon.h5"),
                                            recon_description="MBIRJAX recon of phantom",
                                            alu_description="1 ALU = 0.508 mm")

    # #### Display results
    # change the image data shape to (slices, rows, cols)
    fdk_recon = np.transpose(fdk_recon, axes=(2, 0, 1))
    mbir_recon = np.transpose(mbir_recon, axes=(2, 0, 1))

    vmin = 0
    vmax = downsample_factor[0] * 0.008
    mbirjax.slice_viewer(fdk_recon, data2=mbir_recon, vmin=0, vmax=vmax, slice_axis=0, slice_axis2=0, slice_label='FDK', slice_label2='MBIR', title='Axial Slice')
    mbirjax.slice_viewer(fdk_recon, data2=mbir_recon, vmin=0, vmax=vmax, slice_axis=1, slice_axis2=1, slice_label='FDK', slice_label2='MBIR', title='Coronal Slice')
    mbirjax.slice_viewer(fdk_recon, data2=mbir_recon, vmin=0, vmax=vmax, slice_axis=2, slice_axis2=2, slice_label='FDK', slice_label2='MBIR', title='Sagittal Slice')


    print("\n*******************************************************",
          "\n************ Calculate MAR sinogram weights ***********",
          "\n*******************************************************")
    # #### Put image back in original order and compute MAR weights
    init_recon = np.transpose(mbir_recon, axes=(1, 2, 0))
    weights_mar = ct_model.gen_weights_mar(sino, init_recon=init_recon, beta=1.0, gamma=3.0)

    # #### Perform MBIR reconstruction with MAR weights
    time0 = time.time()
    mbir_mar_recon, mbir_mar_recon_params = ct_model.recon(sino, init_recon=init_recon, weights=weights_mar)
    mbir_mar_recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))

    # #### Print out parameters used in recon
    pprint.pprint(mbir_mar_recon_params._asdict())


    # #### Display results
    # change the image data shape to (slices, rows, cols)
    mbir_mar_recon = np.transpose(mbir_mar_recon, axes=(2, 0, 1))

    vmin = 0
    vmax = downsample_factor[0] * 0.008
    mbirjax.slice_viewer(mbir_recon, data2=mbir_mar_recon, vmin=0, vmax=vmax, slice_axis=0, slice_axis2=0, slice_label='MBIR', slice_label2='MBIR MAR', title='Axial Slice')
    mbirjax.slice_viewer(mbir_recon, data2=mbir_mar_recon, vmin=0, vmax=vmax, slice_axis=1, slice_axis2=1, slice_label='MBIR', slice_label2='MBIR MAR', title='Coronal Slice')
    mbirjax.slice_viewer(mbir_recon, data2=mbir_mar_recon, vmin=0, vmax=vmax, slice_axis=2, slice_axis2=2, slice_label='MBIR', slice_label2='MBIR MAR', title='Sagittal Slice')

