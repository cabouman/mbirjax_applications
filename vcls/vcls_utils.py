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


def copy_ct_model(ct_model, new_angles):
    """
    Create a TomographyModel with the same type and parameters as the given ct_model except with the new input angles
    and a corresponding sinogram shape.

    Args:
        ct_model (TomographyModel): The model to copy.
        new_angles (ndarray of float): 1D vector of projection angles in radians

    Returns:
        An instance of ConeBeamModel or ParallelBeam model
    """
    required_param_names = ct_model.get_required_param_names()
    required_params, other_params = ct_model.get_required_params_from_dict(ct_model.params,
                                                                           required_param_names=required_param_names,
                                                                           values_only=True)

    #  Get the shape of the old sinogram
    old_shape = ct_model.get_params('sinogram_shape')

    # Set the new sinogram shape and angles
    required_params['sinogram_shape'] = (len(new_angles), old_shape[1], old_shape[2])
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



def vcls(ct_model, reference_object, num_selected_views, r_1=0.001, r_2=0.1, verbose=0, seed=None):
    """
    Run the View Correlation Loss Selection (VCLS) algorithm to choose an optimal subset of view angles.

    This function selects a subset of K views that minimize the View Correlation Loss (VCL) using a stochastic greedy optimization algorithm.
    The VCL is defined in the following paper: ???

    Args:
        ct_model (TomographyModel): A CT model instance (e.g., ParallelBeamModel or ConeBeamModel) containing the system geometry and angles.
        reference_object (ndarray): 3D array representing the reference volume (e.g., ground truth).
        num_selected_views (int): Number of view angles to select.
        r_1 (float, optional): Voxel sampling rate in the reference object (default is 0.001).
        r_2 (float, optional): View sampling rate for stochastic minimization (default is 0.01).
        verbose (int, optional): Verbosity level. If > 0, visualizations of the covariance matrix and gamma vector will be shown.
        seed (int, optional): Random seed for deterministic behavior. If set, results will be reproducible.

    Returns:
        ndarray: A 1D NumPy array of the selected optimal view angles of shape (K,).

    Example:
        >>> angles = np.linspace(0, np.pi, num=180, endpoint=False)
        >>> sinogram_shape = (180, 128, 1)
        >>> ct_model = mj.ParallelBeamModel(sinogram_shape, angles)
        >>> ref_obj = np.random.rand(128, 128, 1)
        >>> selected_angles = vcls(ct_model, ref_obj, num_selected_views=10)
        >>> print(selected_angles.shape)
        (10,)
    """
    num_views = ct_model.get_params('sinogram_shape')[0]
    angle_candidates = np.asarray(ct_model.get_params('angles'))
    with tempfile.TemporaryDirectory() as data_store_dir:
        # Compute recon bases
        gamma = compute_recon_bases(ct_model, reference_object, r_1=r_1, data_store_dir=data_store_dir, seed=seed)

        # Compute inner product between recon bases
        R = parallel_cov_matrix_computation(num_views, data_store_dir)

    if verbose > 0:
        # plot the the covariance matrix and gamma
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(R)
        axes[0].set_title('Normalized R')
        axes[1].imshow(np.linalg.inv(R))
        axes[1].set_title('R Inverse')
        axes[2].plot(gamma, '.')
        axes[2].set_ylim([0, np.max(gamma)])
        axes[2].set_title('VCL Gamma vector')
        plt.tight_layout()
        plt.show()

    # Compute optimal view angles
    optimal_angles = angle_subset_selection(R, gamma, angle_candidates, num_selected_views, r_2, seed=seed)
    optimal_angles = np.sort(optimal_angles).flatten()

    return optimal_angles



def compute_recon_bases(ct_model, ref_object, r_1, data_store_dir, seed=None):
    """
    Compute the reconstruction bases and inner product vector (gamma) used in the VCLS algorithm.

    Args:
        ct_model (TomographyModel): CT model specifying the system geometry.
        ref_object (ndarray): 3D reference object with shape (rows, cols, slices).
        r_1 (float): Voxel sampling rate in the reference object (fraction of total voxels).
        data_store_dir (str): Directory where the computed reconstructions will be stored as .npy files.
        seed (int, optional): Random seed for deterministic behavior. Default is None.

    Returns:
        ndarray: A 2D array of shape (num_views, 1) representing the gamma column vector.

    Example:
        >>> gamma = compute_recon_bases(ct_model, ref_object, 0.001, "/tmp/recons")
        >>> print(gamma.shape)
        (180, 1)
    """
    # Define epsilon to avoid divide by zero
    eps = 1e-12

    # Compute forward projection of reference object
    print('Creating sinogram')
    ref_sino = ct_model.forward_project(ref_object)
    ref_sino = np.asarray(ref_sino)

    # Create mask that defines the region of reconstruction (ROR)
    mask = mj.get_2d_ror_mask(ref_object[:, :, 0].shape)
    norm_x = np.linalg.norm((mask[:, :, None] * ref_object).flatten())

    # subsampling voxel indices in ROI
    sparse_indices, row_col_indices = subsampling2d_indices(mask, r_1, seed=seed)
    ref_object_flat = ref_object.reshape(ref_object.shape[0] * ref_object.shape[1], ref_object.shape[2])
    sparse_ref_object = ref_object_flat[sparse_indices, :].flatten()

    # Initialize arrays
    num_views = ct_model.get_params('sinogram_shape')[0]
    candidate_angles = np.asarray(ct_model.get_params('angles'))
    gamma = np.zeros((num_views, 1))  # Inner product between reference object and recon from a single angle

    # Compute recon bases - choose one view at a time and do an fbp/fdk from that.
    print('Creating recon bases')
    for i in tqdm.tqdm(range(num_views)):
        one_angle_sino = ref_sino[[i], :, :]
        one_angle = candidate_angles[i: i + 1]
        one_angle_model = copy_ct_model(ct_model, one_angle)

        # Filter sinogram using appropriate filter for geometry
        filtered_sinogram = one_angle_model.direct_filter(one_angle_sino, view_batch_size=None)

        # Compute normalized sparse reconstruction basis, T_\theta in paper
        sparse_recon_basis = one_angle_model.sparse_back_project(filtered_sinogram, sparse_indices).flatten()
        norm = np.linalg.norm(sparse_recon_basis)
        normalized_sparse_recon_basis = sparse_recon_basis / (norm + eps)

        with open(os.path.join(data_store_dir, f'recon_view{i}.npy'), 'wb') as f:
            np.save(f, normalized_sparse_recon_basis)

        gamma[i, :] = np.sum(normalized_sparse_recon_basis * sparse_ref_object) / (norm_x + eps)

    return gamma


def compute_cov_matrix_part(i, num_views, data_store_dir):
    row = np.zeros(num_views)
    recon_i = np.load(os.path.join(data_store_dir, f'recon_view{i}.npy'))
    for j in range(i, num_views):
        recon_j = np.load(os.path.join(data_store_dir, f'recon_view{j}.npy'))
        row[j] = np.dot(recon_i, recon_j)

    return i, row


def parallel_cov_matrix_computation(num_views, data_store_dir):
    # Set number of processors
    num_cpus = mp.cpu_count()
    print('Number of CPUs: ', num_cpus)

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


def angle_subset_selection(R, gamma, angle_candidates, K, r_2, seed=None):
    """
    Select a subset of view angles that minimize the View Correlation Loss (VCL) using stochastic greedy optimization.

    This function performs an iterative stochastic search over candidate view indices to minimize the VCL,
    defined as VCL = -γᵀ R⁻¹ γ, where R is a covariance matrix of reconstructions and γ is the inner product vector.
    At each step, it considers random replacements of the current selection and keeps changes that improve the loss.

    Args:
        R (ndarray): Covariance matrix of shape (num_views, num_views).
        gamma (ndarray): Column vector of shape (num_views, 1), representing the inner product between reconstructions and reference.
        angle_candidates (ndarray): 1D array of view angles (shape (num_views,)) corresponding to R and gamma.
        K (int): Number of view angles to select.
        r_2 (float): Fraction of unchosen candidates to sample per view per iteration.
        seed (int, optional): Random seed for deterministic behavior. Default is None.

    Returns:
        ndarray: A 1D NumPy array of selected view angles of shape (num_selected_views,).

    Example:
        >>> selected = angle_subset_selection(R, gamma, angle_candidates, num_selected_views=10, r_2=0.01)
        >>> print(selected.shape)
        (10,)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

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





def subsampling2d_indices(mask, r_1, seed=None):
    if seed is not None:
        np.random.seed(seed)
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
