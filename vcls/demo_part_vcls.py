import multiprocessing
seed = 42  # Change this value to control randomness across runs

import numpy as np
import time
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp
import utils as dut
import os


if __name__ == '__main__':

    ##############################################
    # Sets user selectable parameters
    ##############################################

    # Load reference object
    npy_dir = '/depot/bouman/users/lin1311/hexagonal_public_data'
    reference_object = np.load(os.path.join(npy_dir, f'reference_object.npy'))

    # Load views candidate
    angle_candidates = np.load(os.path.join(npy_dir, f'angle_candidates_list.npy'))

    # Set geometry parameters
    geometry_type = 'cone'  # 'cone' or 'parallel'
    num_object_rows = reference_object.shape[0]
    num_object_slices = reference_object.shape[2]
    source_detector_dist = 4042.54
    source_iso_dist = 1216.535
    det_channel_offset = -0.867
    det_row_offset = 0.415
    num_candidate_views = len(angle_candidates)
    start_angle = 0
    end_angle = 2 * np.pi

    #####################
    # Set VCLS parameters
    #####################
    num_selected_views = 40
    # r_1 = voxel sampling rate in (0,1]. Smaller => faster; Larger => more accurate
    r_1 = 0.002
    # r_2 = view sampling rate used for stochastic search in (0,1]. Smaller => faster; Larger => more accurate
    r_2 = 0.5


    ####################################################
    # Calculate function parameters from user parameters
    ####################################################
    # We need this to do multiprocessing in vcls_utils.compute_cov_matrix
    multiprocessing.freeze_support()

    # Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
    # channels, then the generated phantom may not have an interior.
    num_views = num_candidate_views
    num_det_rows = reference_object.shape[2]
    num_det_channels = reference_object.shape[0]
    sinogram_shape = (num_views, num_det_rows, num_det_channels)

    # Create the model to contain all the geometry information
    ct_model = mjp.get_ct_model(geometry_type, sinogram_shape, angle_candidates, source_detector_dist, source_iso_dist)

    ##############################################
    # Run VCLS to Select Views and Display Results
    ##############################################
    time0 = time.time()
    optimal_angles, vcl_value = mjp.get_opt_views(ct_model, reference_object, num_selected_views, r_1=r_1, r_2=r_2, verbose=1, seed=seed)
    elapsed = time.time() - time0
    print('Elapsed time for selected views is {:.3f} seconds'.format(elapsed))
    print('VCL value for selected views: {:.6f}'.format(vcl_value))

    # Display reference object cross-section with selected angles
    formatted = np.array2string(optimal_angles, precision=3, suppress_small=True, separator=', ')
    print('chosen angles: ' + formatted)
    mjp.show_image_with_angles(reference_object[:, :, 0], angles_rad=optimal_angles, title='Reference Object with Selected View Angles')

    # Display reference object Fourier transform along with selected angles
    ref_fft = np.fft.fftshift(np.fft.fft2(reference_object, axes=(0, 1)))[:, :, 4]
    angles_perp = optimal_angles + np.pi / 2    # Add 90deg because Fourier transform of edge is perpendicular to edge
    mjp.show_image_with_angles(np.log10(1e-2 + np.abs(ref_fft)), angles_rad=angles_perp, title='FFT of Reference Object\n with Selected View Angles')

    # Do a recon with optimal angles
    ct_model_opt = mjp.copy_ct_model(ct_model, optimal_angles)
    sinogram_optimal_angles = ct_model_opt.forward_project(reference_object)
    recon_optimal_angles, recon_params = ct_model_opt.recon(sinogram_optimal_angles)

    angles = jnp.linspace(start_angle, end_angle, len(optimal_angles), endpoint=False)
    ct_model_uniform = mjp.copy_ct_model(ct_model, angles)
    sinogram_uniform = ct_model_uniform.forward_project(reference_object)
    recon_uniform, recon_params_uniform = ct_model_uniform.recon(sinogram_uniform)

    mj.slice_viewer(reference_object, recon_uniform, recon_optimal_angles, slice_label=['Ref object', 'Uniform Angles', 'VCLS Angles'],
                    title='Reference object (left) plus Recons from \nuniformly spaced angles (middle) and optimal angles (right)')
