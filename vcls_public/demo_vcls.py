import multiprocessing

import numpy as np
import time
import jax.numpy as jnp
import mbirjax as mj
import _utils as ut
import vcls_utils as vut
import os

import numpy as np
import matplotlib.pyplot as plt

def show_image_with_angles(image: np.ndarray, angles_deg: np.ndarray) -> None:
    """
    Display a square image and overlay lines representing angles.

    Parameters
    ----------
    image : np.ndarray
        A 2D square numpy array representing the image.
    angles_deg : np.ndarray
        A 1D array of angles in degrees. Each angle will be shown as a line
        through the image center in both directions.

    Returns
    -------
    None
    """
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("Image must be a square 2D array")

    side_length = image.shape[0]
    center = side_length / 2
    radius = side_length / 2  # Half-length of the line to reach from center to edge

    # Plot the image
    plt.imshow(image, cmap='gray', origin='upper', extent=[0, side_length, side_length, 0])
    plt.gca().set_aspect('equal')

    # Overlay lines for each angle
    colors = plt.cm.tab10(np.arange(len(angles_deg)) % 10)

    for i, angle_deg in enumerate(angles_deg):
        theta = np.deg2rad(angle_deg)
        dx = radius * np.cos(theta)
        dy = radius * np.sin(theta)

        x0, x1 = center - dx, center + dx
        y0, y1 = center - dy, center + dy

        plt.plot([x0, x1], [y0, y1], color=colors[i], linewidth=2)

    plt.title("Image with Overlaid Angles")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':

    # Define primary parameters
    num_object_rows = 128
    num_object_slices = 64
    num_candidate_views = 128
    num_selected_views = 30
    data_store_dir = f'./recon_bases_data/polygon_data'

    # Do setup
    multiprocessing.freeze_support()
    """**Set the geometry parameters**"""
    # Generate polygon phantom
    print('Creating phantom')
    reference_object = ut.gen_polygon_phantom(num_rows=num_object_rows, num_slices=num_object_slices)
    print(f'reference_object shape: {reference_object.shape}')

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
    angles_candidates = jnp.linspace(start_angle, end_angle, ct_params['num_views'], endpoint=False)

    # vcls parameters
    vcls_parms = {}
    vcls_parms['K'] = num_selected_views # num of selected views
    vcls_parms['r_1'] = 0.001
    vcls_parms['r_2'] = 0.1
    vcls_parms['3d_subsample'] = False # Set to True to enable subsampling of different voxel indices across slices
    #vcls_parms['num_cpus'] = mp.cpu_count()
    vcls_parms['num_cpus'] = 4

    os.makedirs(data_store_dir, exist_ok=True)
    time0 = time.time()

    # #### run vcls to select views ####
    optimal_angles = vut.vcls(reference_object, angles_candidates, ct_params, vcls_parms, data_store_dir)

    # Record elapsed time
    elapsed = time.time() - time0
    print('Elapsed time for selected views is {:.3f} seconds'.format(elapsed))

    # Convert to degrees and display
    angles_arr = jnp.stack(optimal_angles) * 180 / np.pi
    angles_arr = np.sort(angles_arr)
    formatted = np.array2string(angles_arr, precision=3, suppress_small=True, separator=', ')
    print('chosen angles: ' + formatted)

    show_image_with_angles(reference_object[:, :, 0], angles_arr)
