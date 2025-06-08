seed = 42  # Change this value to control randomness across runs

import numpy as np
import time
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp
import utils as dut
import ornl_utils as out
import os

if __name__ == '__main__':

    ##############################################
    # Sets user selectable parameters
    ##############################################

    # ##### params for dataset downloading. User may change these parameters for their own datasets.
    # An example dataset will be downloaded from `dataset_url`, and saved to `download_dir`.
    # url to dataset.
    dataset_url = '/depot/bouman/data/ORNL/hexagonal_public_data.tgz'
    # destination path to download and extract the data and metadata.
    download_dir = './demo_data/'
    # Path to scan directory.
    dataset_dir = mj.download_and_extract_tar(dataset_url, download_dir)

    # Load reference object
    print('Loading reference object')
    reference_object = np.load(os.path.join(dataset_dir, f'reference_object.npy'))
    print('Done')

    #####################
    # Set VCLS parameters
    #####################
    num_selected_views = 40
    # r_1 = voxel sampling rate in (0,1]. Smaller => faster; Larger => more accurate
    r_1 = 0.002
    # r_2 = view sampling rate used for stochastic search in (0,1]. Smaller => faster; Larger => more accurate
    r_2 = 0.5

    #####################
    # Recon parameters
    #####################
    sharpness = 1.0
    snr_db = 35.0
    max_iterations = 20

    #####################
    # Construct model
    #####################
    # Load and preprocess ORNL data
    sino, cone_beam_params, optional_params = out.compute_sino_and_params(dataset_dir)

    # Construct cone beam object using ORNL parameters
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    angle_candidates = cone_beam_params['angles']       # This is probably not the best way to do this

    # Set optional ORNL geometry parameters
    ct_model.set_params(**optional_params)

    ##############################################
    # Run VCLS to Select Views and Display Results
    ##############################################
    time0 = time.time()
    optimal_angle_inds, vcl_value = mjp.get_opt_views(ct_model, reference_object, num_selected_views, r_1=r_1, r_2=r_2, verbose=1, seed=seed)
    optimal_angles = angle_candidates[optimal_angle_inds]
    elapsed = time.time() - time0
    print('Elapsed time for selected views is {:.3f} seconds'.format(elapsed))
    print('VCL value for selected views: {:.6f}'.format(vcl_value))

    # Display reference object cross-section with selected angles
    formatted = np.array2string(optimal_angles, precision=3, suppress_small=True, separator=', ')
    print('chosen angles: ' + formatted)
    mjp.show_image_with_projection_rays(reference_object[:, :, 0], rotation_angles_rad=optimal_angles, title='Reference Object with Selected View Angles')

    # Display reference object Fourier transform along with selected angles
    center_slice = reference_object[:, :, reference_object.shape[2] // 2]
    ref_fft = np.fft.fftshift(np.fft.fft2(center_slice))
    angles_perp = optimal_angles + np.pi / 2    # Add 90deg because Fourier transform of edge is perpendicular to edge
    mjp.show_image_with_projection_rays(np.log10(1e-2 + np.abs(ref_fft)), rotation_angles_rad=angles_perp, title='FFT of Reference Object\n with Selected View Angles')

    # Load measured data
    full_sinogram = np.load(os.path.join(dataset_dir, f'measured_projection.npy'))

    # Update ct_model with new sinogram shape
    ct_model = mjp.get_ct_model(geometry_type, full_sinogram.shape, angle_candidates, source_detector_dist, source_iso_dist)
    ct_model.set_params(det_channel_offset=det_channel_offset, det_row_offset=det_row_offset, sharpness=sharpness, snr_db=snr_db)

    # Do a recon with optimal angles
    optimal_index_list = np.argmin(
        np.abs(angle_candidates[:, None] - optimal_angles[None, :]),
        axis=0
    )
    optimal_angles = angle_candidates[optimal_index_list]
    ct_model_opt = mjp.copy_ct_model(ct_model, optimal_angles)
    sinogram_optimal_angles = full_sinogram[optimal_index_list]
    recon_optimal_angles, recon_params = ct_model_opt.recon(sinogram_optimal_angles, max_iterations=max_iterations)

    # Do a recon with uniform angles
    ct_model_uniform = mjp.copy_ct_model(ct_model, uniform_angles)
    sinogram_uniform = full_sinogram[uniform_index_list]
    recon_uniform, recon_params_uniform = ct_model_uniform.recon(sinogram_uniform, max_iterations=max_iterations)

    mj.slice_viewer(recon_uniform, recon_optimal_angles, slice_label=['Uniform: Slice', 'VCLS optimal: Slice'],
                    title='Recons from {} views: \nuniformly spaced angles (left) and optimal angles (right)'.format(num_selected_views), vmin=0.0, vmax=0.05)

