import numpy as np
import jax.numpy as jnp
import mbirjax as mj
import tqdm  # Included in mbirjax
import _utils as ut
import os
import multiprocessing as mp
import time
import random

def vcls(reference_object,angles_candidates,ct_params,vcls_parms,data_store_dir):

    # Compute recon bases
    time0 = time.time()

    gamma = ComputeReconBases(reference_object,angles_candidates,ct_params,vcls_parms,data_store_dir)

    elapsed = time.time() - time0
    print('Elapsed time for compute recon bases is {:.3f} seconds'.format(elapsed))

    # Compute inner product between recon bases
    time0 = time.time()

    R = parallel_cov_matrix_computation(ct_params['num_views'], vcls_parms['num_cpus'], data_store_dir)

    elapsed = time.time() - time0
    print('Elapsed time for compute inner product is {:.3f} seconds'.format(elapsed))

    # Find optimal view angles
    time0 = time.time()

    optimal_indices = view_subset_selection(R, gamma, ct_params['num_views'], vcls_parms['K'], vcls_parms['r_2'])
    optimal_angles = angles_candidates[optimal_indices]

    elapsed = time.time() - time0
    print('Elapsed time for compute optimal subset of views is {:.3f} seconds'.format(elapsed))

    return optimal_angles


def ComputeReconBases(reference_object,angles_candidates,ct_params,vcls_parms,data_store_dir):

    # Initialize sinogram

    # TODO: Add all necessary parameters to define the CT geometry correctly (e.g., offset, etc.)
    if ct_params['geometry_type'] == 'cone':
        ct_model_for_generation = mj.ConeBeamModel(ct_params['sinogram_shape'], angles_candidates,
                                                   source_detector_dist=ct_params['source_detector_dist'],
                                                   source_iso_dist=ct_params['source_iso_dist'])
    elif ct_params['geometry_type'] == 'parallel':
        ct_model_for_generation = mj.ParallelBeamModel(ct_params['sinogram_shape'], angles_candidates)
    else:
        raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(ct_params['geometry_type']))

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = ct_model_for_generation.forward_project(reference_object)
    sinogram = np.asarray(sinogram)

    # View sinogram
    # title = 'Original sinogram \nUse the sliders to change the view or adjust the intensity range.'
    # mj.slice_viewer(sinogram, slice_axis=0, title=title, slice_label='View')

    # define ROI
    if vcls_parms['3d_subsample']:
        mask = ut.Create3DMask(reference_object)
    else:
        mask = ut.Create2DMask(reference_object[:,:,0])

    # View ROI
    # mj.slice_viewer(reference_object, mask, slice_axis=2, slice_label='View')

    # subsampling voxel indices in ROI
    if vcls_parms['3d_subsample']:
        sub_indices = ut.Subsampling3DIndices(mask,vcls_parms['r_1'])
        phantom_sub_values = reference_object[sub_indices]
    else:
        sub_indices_3d, random_indices_2d, row_col_indices = ut.Subsampling2DIndices(mask, reference_object.shape[2], vcls_parms['r_1'])
        phantom_sub_values = reference_object[sub_indices_3d]

    gamma = np.zeros((ct_params['num_views'], 1))

    # Compute recon bases
    print('Creating recon bases')
    for i in tqdm.tqdm(range(ct_params['num_views'])):
        sinogram_temp = sinogram[[i], :, :]
        angle_temp = angles_candidates[i : i+1]
        cone_model = mj.ConeBeamModel(sinogram_temp.shape, angle_temp,
                                      source_detector_dist=ct_params['source_detector_dist'],
                                      source_iso_dist=ct_params['source_iso_dist'])

        if vcls_parms['3d_subsample']:
            recon_3d = cone_model.fdk_recon(sinogram_temp)
            #recon_3d.block_until_ready()
            rec_sub_values = recon_3d[sub_indices]

        else:
            filtered_sinogram = cone_model.fdk_filter(sinogram_temp, filter_name="ramp", view_batch_size=None)
            recon_cylinder = cone_model.sparse_back_project(filtered_sinogram,random_indices_2d)
            rec_sub_values = recon_cylinder.flatten()
            # #To-do: recon_cylinder does not require to put back to recon_3d after ensure the order is corrsponded
            # recon_3d = jnp.zeros(reference_object.shape)
            # recon_3d = recon_3d.at[row_col_indices].set(recon_cylinder)
            # rec_sub_values = recon_3d[sub_indices_3d]


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
        product = np.multiply(recon_i, recon_j)
        row[j] = np.sum(product)

    return i, row

def parallel_cov_matrix_computation(num_views, num_cpus,data_store_dir):

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

def compute_vcl(sub_R,sub_gamma):

    # beta_transpose = np.transpose(sub_gamma)
    # R_inverse = np.linalg.inv(sub_R)
    # matrix_temp = beta_transpose @ R_inverse
    # loss_value = (matrix_temp @ sub_gamma) * -1

    loss_value = - sub_gamma.T @ np.linalg.solve(sub_R, sub_gamma)
    return loss_value

def view_subset_selection(R,gamma,num_candidate_views,K,r_2):

    max_num_iteration = 100
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

    return indices_chosen