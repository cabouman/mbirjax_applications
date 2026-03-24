seed = 42  # Change this value to control randomness across runs

import numpy as np
import time
import jax.numpy as jnp
import mbirjax as mj
import utils as dut



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
    priority_order = True # reorders the selected view indices from most to least important


    ####################################################
    # Calculate function parameters from user parameters
    ####################################################

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
    ct_model = mj.get_ct_model(geometry_type, sinogram_shape, angle_candidates, source_detector_dist, source_iso_dist)

    ##############################################
    # Run VCLS to Select Views and Display Results
    ##############################################
    time0 = time.time()
    # Optional: Select the best 2 views
    # This can be useful in real application when a small number of views is measured first.
    inital_number_of_views = 2
    prev_selected_view_inds, vcl_value = mj.get_opt_views(ct_model, reference_object, inital_number_of_views, r_1=r_1, r_2=r_2, verbose=1, seed=seed)

    # The select the remaining num_selected_views-2 views
    optimal_angle_inds, vcl_value = mj.get_opt_views(ct_model, reference_object, num_selected_views - inital_number_of_views, r_1=r_1, r_2=r_2, prev_selected_view_inds=prev_selected_view_inds, priority_order=priority_order, verbose=1, seed=seed)
    optimal_angles = angle_candidates[optimal_angle_inds]
    elapsed = time.time() - time0
    print('Elapsed time for selected views is {:.3f} seconds'.format(elapsed))
    print('VCL value for selected views: {:.6f}'.format(vcl_value))

    # Display reference object cross-section with selected angles
    optimal_angles = jnp.concatenate([prev_selected_angles, optimal_angles])
    formatted = np.array2string(optimal_angles, precision=3, suppress_small=True, separator=', ')
    print('chosen angles: ' + formatted)
    mj.show_image_with_projection_rays(reference_object[:, :, 0], rotation_angles_rad=optimal_angles, title='Reference Object with Selected View Angles')

    # Display reference object Fourier transform along with selected angles
    ref_fft = np.fft.fftshift(np.fft.fft2(reference_object, axes=(0, 1)))[:, :, 4]
    angles_perp = optimal_angles + np.pi / 2    # Add 90deg because Fourier transform of edge is perpendicular to edge
    mj.show_image_with_projection_rays(np.log10(1e-2 + np.abs(ref_fft)), rotation_angles_rad=angles_perp, title='FFT of Reference Object\n with Selected View Angles')

    # Do a recon with optimal angles
    ct_model_opt = mj.copy_ct_model(ct_model, optimal_angles)
    sinogram_optimal_angles = ct_model_opt.forward_project(reference_object)
    recon_optimal_angles, recon_dict_optimal = ct_model_opt.recon(sinogram_optimal_angles)

    angles = jnp.linspace(start_angle, end_angle, len(optimal_angles), endpoint=False)
    ct_model_uniform = mj.copy_ct_model(ct_model, angles)
    sinogram_uniform = ct_model_uniform.forward_project(reference_object)
    recon_uniform, recon_dict_uniform = ct_model_uniform.recon(sinogram_uniform)

    mj.slice_viewer(reference_object, recon_uniform, recon_optimal_angles,
                    data_dicts=[None, recon_dict_uniform, recon_dict_optimal],
                    slice_label=['Ref object', 'Uniform Angles', 'VCLS Angles'],
                    title='Reference object (left) plus Recons from \nuniformly spaced angles (middle) and optimal angles (right)')
