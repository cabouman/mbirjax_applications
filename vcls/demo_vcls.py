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
    num_object_rows = 128
    num_object_slices = 64
    num_candidate_views = 128
    num_selected_views = 25

    # Cone-beam geometry parameters
    magnification = 2.0
    cone_angle = (15/180)*np.pi     # cone angle in radians:  50/180 corresponds to 50 degrees

    # Set vcls algorithm parameters
    voxel_sampling_rate = 0.01      # r_1 in paper
    view_sampling_rate = 0.1        # r_2 in paper

    # Set view angle limits
    start_angle = 0
    end_angle = 2 * np.pi

    ####################################################
    # Calculate function parameters from user parameters
    ####################################################

    multiprocessing.freeze_support()  # We need this to do multiprocessing in vcls_utils.parallel_cov_matrix_computation

    # Create reference object
    print('Creating phantom')
    reference_object = dut.gen_polygon_phantom(num_rows=num_object_rows, num_slices=num_object_slices)
    print(f'reference_object shape: {reference_object.shape}')

    # Setup ct_params values
    geometry_type = 'cone'  # 'cone' or 'parallel'

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

    # Set vcls parameters
    vcls_params = {}
    vcls_params['K'] = num_selected_views # num of selected views
    vcls_params['r_1'] = voxel_sampling_rate
    vcls_params['r_2'] = view_sampling_rate
    vcls_params['3d_subsample'] = False # Set to True to enable subsampling of different voxel indices across slices
    #vcls_params['num_cpus'] = mp.cpu_count()
    vcls_params['num_cpus'] = 4

    time0 = time.time()

    ##############################################
    # Run VCLS to Select Views and Display Results
    ##############################################

    # #### run vcls to select views ####
    optimal_angles = vut.vcls(reference_object, ct_model, vcls_params)

    # Record elapsed time
    elapsed = time.time() - time0
    print('Elapsed time for selected views is {:.3f} seconds'.format(elapsed))

    # Convert to degrees and display
    angles_arr = jnp.stack(optimal_angles) * 180 / np.pi
    angles_arr = np.sort(angles_arr).flatten()
    formatted = np.array2string(angles_arr, precision=3, suppress_small=True, separator=', ')
    print('chosen angles: ' + formatted)

    # Display selected angles
    dut.show_image_with_angles(reference_object[:, :, 0], angles_rad=angles_arr)

    # Display the default and optimal angle recons
    new_num_views = len(angles_arr)
    sinogram_shape = (new_num_views, sinogram_shape[1], sinogram_shape[2])

    # Do a recon with optimal angles
    ct_model = vut.copy_ct_model(ct_model, sinogram_shape, angles_arr)
    sinogram_optimal_angles = ct_model.forward_project(reference_object)
    recon_optimal_angles, recon_params = ct_model.recon(sinogram_optimal_angles)

    angles = jnp.linspace(start_angle, end_angle, new_num_views, endpoint=False)
    ct_model = vut.copy_ct_model(ct_model, sinogram_shape, angles)
    sinogram_uniform = ct_model.forward_project(reference_object)
    recon_uniform, recon_params_uniform = ct_model.recon(sinogram_uniform)

    mj.slice_viewer(reference_object, recon_uniform, recon_optimal_angles, slice_label=['Ref object', 'Uniform Angles', 'VCLS Angles'],
                    title='Reference object (left) plus Recons from \nuniformly spaced angles (middle) and optimal angles (right)')
