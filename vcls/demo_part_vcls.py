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

    # Load angle candidates
    angle_candidates = np.load(os.path.join(dataset_dir, f'angle_candidates_list.npy'))

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

    #####################
    # Recon parameters
    #####################
    sharpness = 1.0
    snr_db = 35.0
    max_iterations = 20


    ####################################################
    # Calculate function parameters from user parameters
    ####################################################

    # Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
    # channels, then the generated phantom may not have an interior.
    num_views = num_candidate_views
    num_det_rows = reference_object.shape[2]
    num_det_channels = reference_object.shape[0]
    sinogram_shape = (num_views, num_det_rows, num_det_channels)

    detector_cone_angle = 2 * np.arctan2(num_det_channels / 2, source_detector_dist)
    candidates_normalized = np.abs(angle_candidates - angle_candidates[0])
    end_index = np.where(candidates_normalized < np.pi + detector_cone_angle)[0][-1] # final angle in the short-scan range
    uniform_index_list = dut.create_uniform_index(angle_candidates, end_index, num_selected_views)
    uniform_angles = angle_candidates[uniform_index_list]

    # Create the model to contain all the geometry information
    ct_model = mjp.get_ct_model(geometry_type, sinogram_shape, angle_candidates, source_detector_dist, source_iso_dist)
    ct_model.set_params(det_channel_offset=det_channel_offset, det_row_offset=det_row_offset)

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
    mjp.show_image_with_angles(reference_object[:, :, 0], angles_rad=optimal_angles, title='Reference Object with Selected View Angles')

    # Display reference object Fourier transform along with selected angles
    center_slice = reference_object[:, :, reference_object.shape[2] // 2]
    ref_fft = np.fft.fftshift(np.fft.fft2(center_slice))
    angles_perp = optimal_angles + np.pi / 2    # Add 90deg because Fourier transform of edge is perpendicular to edge
    mjp.show_image_with_angles(np.log10(1e-2 + np.abs(ref_fft)), angles_rad=angles_perp, title='FFT of Reference Object\n with Selected View Angles')

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

