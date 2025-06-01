import os
import multiprocessing as mp
import random
import tempfile
import warnings

import numpy as np
import mbirjax as mj
import jax.numpy as jnp
import tqdm  # Included in mbirjax


def get_ct_model(geometry_type, sinogram_shape, angles, source_detector_dist=None, source_iso_dist=None):
    """
    Create an instance of TomographyModel with the given parameters

    Args:
        geometry_type (str): 'parallel' or 'cone'
        sinogram_shape (tuple list of int): (num_views, num_rows, num_channels)
        angles (ndarray of float): 1D vector of projection angles in radians
        source_detector_dist (float or None, optional): Distance in ALU from source to detector.  Defaults to None for geometries that don't need this.
        source_iso_dist (float or None, optional): Distance in ALU from source to iso.  Defaults to None for geometries that don't need this.

    Returns:
        An instance of ConeBeamModel or ParallelBeam model
    """
    if geometry_type == 'cone':
        model = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                                 source_iso_dist=source_iso_dist)
    elif geometry_type == 'parallel':
        model = mj.ParallelBeamModel(sinogram_shape, angles)
    else:
        raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(geometry_type))

    return model


def copy_ct_model(ct_model, new_sinogram_shape, new_angles):
    """
    Create a TomographyModel with the same type and parameters as the given ct_model except with the input sinogram
    shape and angles.

    Args:
        ct_model (TomographyModel): The model to copy.
        new_sinogram_shape (tuple list of int): (num_views, num_rows, num_channels)
        new_angles (ndarray of float): 1D vector of projection angles in radians

    Returns:
        An instance of ConeBeamModel or ParallelBeam model
    """
    required_param_names = ct_model.get_required_param_names()
    required_params, other_params = ct_model.get_required_params_from_dict(ct_model.params,
                                                                           required_param_names=required_param_names,
                                                                           values_only=True)
    required_params['sinogram_shape'] = new_sinogram_shape
    required_params['angles'] = new_angles
    new_model = type(ct_model)(**required_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_model.set_params(**other_params)

    return new_model


def max_abs_neighbor_diff(arr):
    padded = np.pad(arr, pad_width=1, mode='reflect')
    center = arr
    max_diff = np.zeros_like(arr)

    # Define the directional offsets: (di, dj)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    for di, dj in directions:
        neighbor = padded[1 + di : 1 + di + arr.shape[0],
                          1 + dj : 1 + dj + arr.shape[1]]
        diff = np.abs(center - neighbor)
        np.maximum(max_diff, diff, out=max_diff)

    return max_diff


def vcls(reference_object, ct_model, vcls_params):
    num_views = ct_model.get_params('sinogram_shape')[0]
    angle_candidates = np.asarray(ct_model.get_params('angles'))
    with tempfile.TemporaryDirectory() as data_store_dir:
        # Compute recon bases
        gamma = compute_recon_bases(reference_object, ct_model, vcls_params, data_store_dir)

        # Compute inner product between recon bases
        R = parallel_cov_matrix_computation(num_views, vcls_params['num_cpus'], data_store_dir)

    # Find optimal view angles
    optimal_angles = angle_subset_selection(R, gamma, angle_candidates, vcls_params['K'], vcls_params['r_2'])

    return optimal_angles


def compute_recon_bases(reference_object, ct_model, vcls_params, data_store_dir):
    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = ct_model.forward_project(reference_object)
    sinogram = np.asarray(sinogram)

    # View sinogram
    # title = 'Original sinogram \nUse the sliders to change the view or adjust the intensity range.'
    # mj.slice_viewer(sinogram, slice_axis=0, title=title, slice_label='View')

    # define ROI
    if vcls_params['3d_subsample']:
        mask = create3d_mask(reference_object)
    else:
        mask = create2d_mask(reference_object[:, :, 0])

    # View ROI
    # mj.slice_viewer(reference_object, mask, slice_axis=2, slice_label='View')

    # subsampling voxel indices in ROI
    if vcls_params['3d_subsample']:
        sub_indices = subsampling3d_indices(mask, vcls_params['r_1'])
        phantom_sub_values = reference_object[sub_indices]
    else:
        random_indices_2d, row_col_indices = subsampling2d_indices(mask, vcls_params['r_1'])
        ref_flat = reference_object.reshape(reference_object.shape[0] * reference_object.shape[1], reference_object.shape[2])
        phantom_sub_values = ref_flat[random_indices_2d, :].flatten()

    num_views = ct_model.get_params('sinogram_shape')[0]
    angle_candidates = np.asarray(ct_model.get_params('angles'))
    gamma = np.zeros((num_views, 1))  # Inner product between reference object and recon from a single angle

    # Compute recon bases - choose one view at a time and do an fbp/fdk from that.
    print('Creating recon bases')
    for i in tqdm.tqdm(range(num_views)):
        one_angle_sinogram = sinogram[[i], :, :]
        one_angle = angle_candidates[i: i + 1]
        one_angle_model = copy_ct_model(ct_model, one_angle_sinogram.shape, one_angle)

        if vcls_params['3d_subsample']:
            recon_3d = one_angle_model.direct_recon(one_angle_sinogram)
            rec_sub_values = recon_3d[sub_indices]

        else:
            filtered_sinogram = one_angle_model.direct_filter(one_angle_sinogram, filter_name="ramp",
                                                              view_batch_size=None)
            recon_cylinder = one_angle_model.sparse_back_project(filtered_sinogram, random_indices_2d)
            rec_sub_values = recon_cylinder.flatten()

        #view recon bases
        #mj.slice_viewer(reference_object, recon_3d, slice_axis=2, slice_label='View')

        with open(os.path.join(data_store_dir, f'recon_view{i}.npy'), 'wb') as f:
            np.save(f, rec_sub_values)

        gamma[i, :] = np.sum(rec_sub_values * phantom_sub_values)

    return gamma


def compute_cov_matrix_part(i, num_views, data_store_dir):
    row = np.zeros(num_views)
    recon_i = np.load(os.path.join(data_store_dir, f'recon_view{i}.npy'))
    for j in range(i, num_views):
        recon_j = np.load(os.path.join(data_store_dir, f'recon_view{j}.npy'))
        row[j] = np.dot(recon_i, recon_j)

    return i, row


def parallel_cov_matrix_computation(num_views, num_cpus, data_store_dir):
    cov_matrix = np.zeros((num_views, num_views))

    # Create a pool of workers
    with mp.Pool(processes=num_cpus) as pool:
        results = [pool.apply_async(compute_cov_matrix_part, args=(i, num_views, data_store_dir)) for i in
                   range(num_views)]

        for result in results:
            i, row = result.get()
            cov_matrix[i, i:] = row[i:]
            cov_matrix[i:, i] = row[i:]

    return cov_matrix


def compute_vcl(sub_R, sub_gamma):
    loss_value = - sub_gamma.T @ np.linalg.solve(sub_R, sub_gamma)

    return loss_value


def angle_subset_selection(R, gamma, angle_candidates, K, r_2):
    max_num_iteration = 100
    num_candidate_views = len(angle_candidates)
    num_candidates = int(r_2 * (num_candidate_views - K))
    if num_candidates < 5:
        num_candidates = 5

    # Initialize choosing indices (uniformly sampling)
    step_size = num_candidate_views / K
    indices_chosen = []
    for i in range(K):
        indices_chosen.append(int(step_size * i))
    indices_chosen = np.array(indices_chosen)

    # Subsample the matrix using the uniformly spaced indices
    R_chosen = R[indices_chosen[:, None], indices_chosen]
    gamma_chosen = gamma[indices_chosen, :]

    vcl_target = compute_vcl(R_chosen, gamma_chosen)

    for i in range(max_num_iteration):
        prev_indices_chosen = np.copy(indices_chosen)
        for j in range(K):
            candidate_indices = list(set(range(num_candidate_views)) - set(indices_chosen))
            random.shuffle(candidate_indices)
            candidate_indices = candidate_indices[:num_candidates]

            for k in candidate_indices:
                indices_temp = np.copy(indices_chosen)
                indices_temp[j] = k
                R_temp = R[indices_temp[:, None], indices_temp]
                gamma_temp = gamma[indices_temp, :]
                vcl_temp = compute_vcl(R_temp, gamma_temp)

                if vcl_temp < vcl_target:
                    vcl_target = np.copy(vcl_temp)
                    indices_chosen = np.copy(indices_temp)

        # Early stopping: Check if the indices have changed
        if np.array_equal(indices_chosen, prev_indices_chosen):
            print(f'Early stopping at iteration {i}, no change in indices')
            break

    return angle_candidates[indices_chosen]


def create2d_mask(cur_slice):
    y_indices, x_indices = np.where(cur_slice > 0)

    # Calculate x_min, x_max, y_min, y_max
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # Calculate the center of the circle
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Calculate the radius of the circle as the maximum distance from the center
    radius = np.max(np.sqrt((x_indices - x_center) ** 2 + (y_indices - y_center) ** 2))
    radius_bigger = 1.01 * radius

    # Generate the mask: if the distance from the center is less than the radius, set value to 1
    h, w = cur_slice.shape
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]

    # Compute squared distances
    dist2 = (x - x_center) ** 2 + (y - y_center) ** 2

    # Build boolean mask in one shot
    mask = (dist2 <= radius_bigger ** 2).astype(np.float32)

    return mask


def create3d_mask(phantom, repeat=False):
    if repeat:
        mask2d = create2d_mask(phantom[:, :, 0])
        mask = np.repeat(mask2d[:, :, np.newaxis], phantom.shape[2], axis=2)

    else:
        mask = np.zeros(phantom.shape)
        for i in range(phantom.shape[2]):
            mask[:, :, i] = create2d_mask(phantom[:, :, i])

    return mask


def subsampling3d_indices(mask, r_1):
    num_rows, num_cols, num_slices = mask.shape
    num_samples = int(num_rows * num_cols * r_1)

    random_indices = []
    for slice_idx in range(num_slices):
        # Randomly select unique indices for this slice
        mask_indices = np.where(mask[:, :, slice_idx] == 1)  # Get 2D indices where mask == 1
        # Ensure num_samples does not exceed the number of available points
        if num_samples > len(mask_indices[0]):
            num_samples_temp = len(mask_indices[0])
        else:
            num_samples_temp = num_samples
        slice_choice = np.random.choice(len(mask_indices[0]), num_samples_temp, replace=False)
        row_indices = mask_indices[0][slice_choice]
        col_indices = mask_indices[1][slice_choice]
        random_indices.append((row_indices, col_indices, slice_idx * np.ones(num_samples_temp, dtype=int)))

    # Convert to a single index array for advanced indexing
    random_indices = tuple(np.concatenate(idx) for idx in zip(*random_indices))

    return random_indices


def subsampling2d_indices(mask, r_1):
    num_rows, num_cols = mask.shape
    num_samples = int(num_rows * num_cols * r_1)
    mask_indices = np.where(mask[:, :] == 1)  # Get 2D indices where mask == 1
    # Ensure num_samples does not exceed the number of available points
    if num_samples > len(mask_indices[0]):
        num_samples = len(mask_indices[0])
    slice_choice = np.random.choice(len(mask_indices[0]), num_samples, replace=False)
    row_inds = mask_indices[0][slice_choice]
    col_inds = mask_indices[1][slice_choice]
    random_indices_2d = row_inds * num_cols + col_inds
    random_indices_2d = jnp.array(random_indices_2d)

    return random_indices_2d, (row_inds, col_inds)
