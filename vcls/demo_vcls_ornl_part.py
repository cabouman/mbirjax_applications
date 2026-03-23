seed = 42  # Change this value to control randomness across runs

import numpy as np
import time
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp
import utils as dut
import os

if __name__ == '__main__':
    #######################
    # Sets pointers to data
    #######################

    # Please cite the following source when using this data:
    #     Amir Ziabari, Obaidullah Rahman,  Singanallur Venkatakrishnan, and Ryan Dehoff,
    #     “X-ray Computed Tomography Data of Dense Metallic Components”,
    #     10.13139/ORNLNCCS/2568789, release data: June 2025
    # This work was carried out at Oak Ridge National Laboratory,
    # managed by UT - Battelle, LLC for the U.S.Department of Energy under contract DE-AC05-00OR22725.

    # path or URL to CT scan in h5 format with tgz wrapper
    dataset_url_scan = 'https://www.datadepot.rcac.purdue.edu/bouman/data/hfn_scan.tgz'
    # path or URL to reference object in npy format with tgz wrapper
    dataset_url_reference = 'https://www.datadepot.rcac.purdue.edu/bouman/data/hfn_reference_object.tgz'
    # path to directory for storage of data
    download_dir = './demo_data/'

    #####################
    # Set VCLS parameters
    #####################
    num_selected_views = 40
    priority_order = True  # reorders the selected view indices from most to least important

    ###############
    # Download Data
    ###############
    # Download and extract data
    dataset_dir_scan = mj.download_and_extract(dataset_url_scan, download_dir)
    dataset_dir_reference = mj.download_and_extract(dataset_url_reference, download_dir)

    # Load reference object into workspace
    print('Loading reference object')
    reference_object = np.load(os.path.join(dataset_dir_reference, f'hfn_reference_object.npy'))
    print('Shape of reference object: {}'.format(reference_object.shape))

    #################
    # Construct model
    #################
    # Load and preprocess ORNL data
    # List all files ending in .h5 or .hdf5
    hdf5_files = sorted(
        f for f in os.listdir(dataset_dir_scan)
        if f.lower().endswith(('.h5', '.hdf5'))
    )
    filename = os.path.join(dataset_dir_scan, hdf5_files[0])
    print('Loading sinogram and computing beam hardening correction')
    full_sino, cone_beam_params, optional_params = mjp.pymbir.compute_sino_and_params(filename)
    print('Sinogram shape: {}'.format(full_sino.shape))
    angle_candidates = cone_beam_params['angles']  # This is probably not the best way to do this

    # Construct cone beam object using ORNL parameters
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    # Set optional geometry parameters
    ct_model.set_params(**optional_params)

    ## Force consistency between recon and reference object shapes
    # Print out default recon shape
    recon_shape = ct_model.get_params("recon_shape")
    print('Default reconstruction shape: {}'.format(recon_shape))
    # Set recon shape to reference object shape
    ct_model.set_params(recon_shape=reference_object.shape)

    ##############################################
    # Run VCLS to Select Views and Display Results
    ##############################################
    time0 = time.time()
    optimal_angle_inds, vcl_value = mj.get_opt_views(ct_model, reference_object, num_selected_views, priority_order=priority_order, verbose=1, seed=seed)
    optimal_angles = angle_candidates[optimal_angle_inds]
    elapsed = time.time() - time0
    print('optimal_angle_inds: ', optimal_angle_inds)
    print('Elapsed time for VCLS view selection is {:.3f} seconds'.format(elapsed))
    print('VCL value for selected views = {:.6f}'.format(vcl_value))

    # Display reference object cross-section with selected angles
    formatted = np.array2string(optimal_angles, precision=3, suppress_small=True, separator=', ')
    print('chosen angles: ' + formatted)
    middle_index = reference_object.shape[2] // 2 + 2
    mj.show_image_with_projection_rays(reference_object[:, :, middle_index], rotation_angles_rad=optimal_angles, title='Reference Object with Selected View Angles')

    # Display reference object Fourier transform along with selected angles
    center_slice = reference_object[:, :, reference_object.shape[2] // 2]
    ref_fft = np.fft.fftshift(np.fft.fft2(center_slice))
    angles_perp = optimal_angles + np.pi / 2    # Add 90deg because Fourier transform of edge is perpendicular to edge
    mj.show_image_with_projection_rays(np.log10(1e-2 + np.abs(ref_fft)), rotation_angles_rad=angles_perp, title='FFT of Reference Object\n with Selected View Angles')

    # Compute mbir recon with optimal angles
    optimal_angles = angle_candidates[optimal_angle_inds]
    ct_model_opt = mj.copy_ct_model(ct_model, optimal_angles)
    sinogram_opt = full_sino[optimal_angle_inds]
    recon_opt, recon_dict_opt = ct_model_opt.recon(sinogram_opt)

    # Compute detector cone angle
    num_det_channels_for_recon = cone_beam_params["sinogram_shape"][2]
    source_detector_dist = cone_beam_params["source_detector_dist"]
    detector_cone_angle = 2 * np.arctan2(num_det_channels_for_recon / 2, source_detector_dist)

    # Compute uniform sampling angles for a short scan
    candidates_normalized = np.abs(angle_candidates - angle_candidates[0])
    end_index = np.where(candidates_normalized < np.pi + detector_cone_angle)[0][-1] # final angle in the short-scan range
    uniform_index_list = dut.create_uniform_index(angle_candidates, end_index, num_selected_views)
    uniform_angles = angle_candidates[uniform_index_list]

    # Compute mbir recon for uniformly sampled short scan
    ct_model_uniform = mj.copy_ct_model(ct_model, uniform_angles)
    sino_uniform = full_sino[uniform_index_list]
    recon_uniform, recon_dict_uniform = ct_model_uniform.recon(sino_uniform)

    mj.slice_viewer(recon_uniform, recon_opt, data_dicts=[recon_dict_uniform, recon_dict_opt], slice_label=['Uniform: Slice', 'VCLS optimal: Slice'],
                    title='Recons from {} views: \nuniformly spaced angles (left) and optimal angles (right)'.format(num_selected_views), vmin=0.0, vmax=0.05)
