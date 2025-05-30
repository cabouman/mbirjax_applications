import multiprocessing

import numpy as np
import time
import jax.numpy as jnp
import mbirjax as mj
import demo_utils as dut
import vcls_utils as vut
import os

import numpy as np


if __name__ == '__main__':

    # Define path to vcls temporary scratch space
    data_store_dir = f'./recon_bases_data/polygon_data'

    # Define primary parameters
    num_object_rows = 128
    num_object_slices = 64
    num_candidate_views = 128
    num_selected_views = 30

    # Do setup
    multiprocessing.freeze_support()
    """**Set the geometry parameters**"""
    # Generate polygon phantom
    print('Creating phantom')
    reference_object = dut.gen_polygon_phantom(num_rows=num_object_rows, num_slices=num_object_slices)
    print(f'reference_object shape: {reference_object.shape}')

    ###########################################
    # Set ct_params values
    ###########################################
    ct_params = {}
    # Choose the geometry type
    ct_params['geometry_type'] = 'cone'  # 'cone' or 'parallel'

    # Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
    # channels, then the generated phantom may not have an interior.
    ct_params['num_views'] = num_candidate_views
    ct_params['num_det_rows'] = reference_object.shape[2]
    ct_params['num_det_channels'] = reference_object.shape[0]
    ct_params['sinogram_shape'] = (ct_params['num_views'], ct_params['num_det_rows'], ct_params['num_det_channels'])

    # For cone beam geometry, we need to describe the distances source to detector and source to rotation axis.
    # np.Inf is an allowable value, in which case this is essentially parallel beam
    ct_params['source_detector_dist'] = 4 * ct_params['num_det_channels']
    ct_params['source_iso_dist'] = ct_params['source_detector_dist'] / 2

    start_angle = 0
    end_angle = 2 * np.pi
    angle_candidates = jnp.linspace(start_angle, end_angle, ct_params['num_views'], endpoint=False)

    # vcls parameters
    vcls_params = {}
    vcls_params['K'] = num_selected_views # num of selected views
    vcls_params['r_1'] = 0.01
    vcls_params['r_2'] = 0.1
    vcls_params['3d_subsample'] = False # Set to True to enable subsampling of different voxel indices across slices
    #vcls_params['num_cpus'] = mp.cpu_count()
    vcls_params['num_cpus'] = 4

    os.makedirs(data_store_dir, exist_ok=True)
    time0 = time.time()

    # #### run vcls to select views ####
    optimal_angles = vut.vcls(reference_object, angle_candidates, ct_params, vcls_params, data_store_dir)

    # Record elapsed time
    elapsed = time.time() - time0
    print('Elapsed time for selected views is {:.3f} seconds'.format(elapsed))

    # Convert to degrees and display
    angles_arr = jnp.stack(optimal_angles) * 180 / np.pi
    angles_arr = np.sort(angles_arr)
    formatted = np.array2string(angles_arr, precision=3, suppress_small=True, separator=', ')
    print('chosen angles: ' + formatted)

    dut.show_image_with_angles(reference_object[:, :, 0], angles_arr)
