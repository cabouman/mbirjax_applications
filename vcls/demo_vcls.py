import multiprocessing

import numpy as np
import time
import jax.numpy as jnp
import mbirjax as mj
import demo_utils as dut
import vcls_utils as vut

import numpy as np


if __name__ == '__main__':

    ##############################################
    # Sets user selectable parameters
    ##############################################

    # Set geometry parameters
    geometry_type = 'cone'  # 'cone' or 'parallel'
    num_object_rows = 128
    num_object_slices = 64
    magnification = 2.0
    cone_angle = (15/180)*np.pi     # cone angle in radians:  50/180 corresponds to 50 degrees
    num_candidate_views = 128
    start_angle = 0
    end_angle = 2 * np.pi

    #####################
    # Set VCLS parameters
    #####################
    num_selected_views = 25
    # r_1 = voxel sampling rate in (0,1]. Smaller => faster; Larger => more accurate
    r_1 = 0.01
    # r_2 = view sampling rate used for stochastic search in (0,1]. Smaller => faster; Larger => more accurate
    r_2 = 0.5
    fast = True    # Use built in sparse back projection in mbirjax to speed algorithm


    ####################################################
    # Calculate function parameters from user parameters
    ####################################################
    # We need this to do multiprocessing in vcls_utils.parallel_cov_matrix_computation
    multiprocessing.freeze_support()

    # Create reference object
    print('Creating phantom')
    reference_object = dut.gen_polygon_phantom(num_rows=num_object_rows, num_slices=num_object_slices)
    print(f'reference_object shape: {reference_object.shape}')

    # Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
    # channels, then the generated phantom may not have an interior.
    num_views = num_candidate_views
    num_det_rows = reference_object.shape[2]
    num_det_channels = reference_object.shape[0]
    sinogram_shape = (num_views, num_det_rows, num_det_channels)

    # For cone beam geometry, we need to describe the distances source to detector and source to rotation axis.
    # np.Inf is an allowable value, in which case this is essentially parallel beam
    source_detector_dist = (1.0/np.tan(cone_angle/2.0)) * (num_det_channels/2)
    source_iso_dist = source_detector_dist / magnification

    # Compute view angles
    angle_candidates = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # Create the model to contain all the geometry information
    ct_model = vut.get_ct_model(geometry_type, sinogram_shape, angle_candidates, source_detector_dist, source_iso_dist)

    ##############################################
    # Run VCLS to Select Views and Display Results
    ##############################################
    time0 = time.time()
    optimal_angles = vut.vcls(ct_model, reference_object, K=num_selected_views, r_1=r_1, r_2=r_2, fast=fast, verbose=1)
    elapsed = time.time() - time0
    print('Elapsed time for selected views is {:.3f} seconds'.format(elapsed))

    # Convert to degrees and display
    angles_arr = np.sort(optimal_angles).flatten()
    formatted = np.array2string(angles_arr, precision=3, suppress_small=True, separator=', ')
    print('chosen angles: ' + formatted)

    # Display selected angles
    dut.show_image_with_angles(reference_object[:, :, 0], angles_rad=angles_arr)

    # Display the default and optimal angle recons
    new_num_views = len(angles_arr)
    sinogram_shape = (new_num_views, sinogram_shape[1], sinogram_shape[2])

    # Do a recon with optimal angles
    optimal_angles = np.sort(jnp.stack(optimal_angles)).flatten()
    ct_model = vut.copy_ct_model(ct_model, sinogram_shape, optimal_angles)
    sinogram_optimal_angles = ct_model.forward_project(reference_object)
    recon_optimal_angles, recon_params = ct_model.recon(sinogram_optimal_angles)

    angles = jnp.linspace(start_angle, end_angle, new_num_views, endpoint=False)
    ct_model = vut.copy_ct_model(ct_model, sinogram_shape, angles)
    sinogram_uniform = ct_model.forward_project(reference_object)
    recon_uniform, recon_params_uniform = ct_model.recon(sinogram_uniform)

    mj.slice_viewer(reference_object, recon_uniform, recon_optimal_angles, slice_label=['Ref object', 'Uniform Angles', 'VCLS Angles'],
                    title='Reference object (left) plus Recons from \nuniformly spaced angles (middle) and optimal angles (right)')
