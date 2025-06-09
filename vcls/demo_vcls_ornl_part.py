seed = 42  # Change this value to control randomness across runs

import numpy as np
import time
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp
import utils as dut
import ornl_utils as out
import os

if __name__ == '__main__':
    #######################
    # Sets pointers to data
    #######################
    # path or URL to CT scan in h5 format with tgz wrapper
    dataset_url_scan = '/depot/bouman/data/ORNL/hfn_scan.tgz'
    # path or URL to reference object in npy format with tgz wrapper
    dataset_url_reference = '/depot/bouman/data/ORNL/hfn_reference_object.tgz'
    # path to directory for storage of data
    download_dir = './demo_data/'

    #####################
    # Set VCLS parameters
    #####################
    num_selected_views = 40

    ######################
    # Set recon parameters
    ######################
    sharpness = 1.0
    snr_db = 35.0
    max_iterations = 20


    ###############
    # Download Data
    ###############
    # Download and extract data
    dataset_dir_scan = mj.download_and_extract_tar(dataset_url_scan, download_dir)
    dataset_dir_reference = mj.download_and_extract_tar(dataset_url_reference, download_dir)

    # Load reference object into workspace
    print('Loading reference object')
    reference_object = np.load(os.path.join(dataset_dir_reference, f'reference_object.npy'))
    reference_obj_shape = reference_object.shape
    print('Shape of reference object: {}'.format(reference_obj_shape))

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
    full_sino, cone_beam_params, optional_params = out.compute_sino_and_params(filename)
    angle_candidates = cone_beam_params['angles']  # This is probably not the best way to do this

    # Construct cone beam object using ORNL parameters
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    # Set optional geometry parameters
    ct_model.set_params(**optional_params)

    # Print out recon shape
    recon_shape = ct_model.get_params("recon_shape")
    print('Default reconstruction shape: {}'.format(recon_shape))
    ct_model.set_params(recon_shape=reference_object)

    ##############################################
    # Run VCLS to Select Views and Display Results
    ##############################################
    time0 = time.time()
    optimal_angle_inds, vcl_value = mjp.get_opt_views(ct_model, reference_object, num_selected_views, verbose=1, seed=seed)
    optimal_angles = angle_candidates[optimal_angle_inds]
    elapsed = time.time() - time0
    print('Elapsed time for VCLS view selection is {:.3f} seconds'.format(elapsed))
    print('VCL value for selected views = {:.6f}'.format(vcl_value))

    # Display reference object cross-section with selected angles
    formatted = np.array2string(optimal_angles, precision=3, suppress_small=True, separator=', ')
    print('chosen angles: ' + formatted)
    mjp.show_image_with_projection_rays(reference_object[:, :, 0], rotation_angles_rad=optimal_angles, title='Reference Object with Selected View Angles')

    # Display reference object Fourier transform along with selected angles
    center_slice = reference_object[:, :, reference_object.shape[2] // 2]
    ref_fft = np.fft.fftshift(np.fft.fft2(center_slice))
    angles_perp = optimal_angles + np.pi / 2    # Add 90deg because Fourier transform of edge is perpendicular to edge
    mjp.show_image_with_projection_rays(np.log10(1e-2 + np.abs(ref_fft)), rotation_angles_rad=angles_perp, title='FFT of Reference Object\n with Selected View Angles')

    # Compute mbir recon with optimal angles
    optimal_angles = angle_candidates[optimal_angle_inds]
    ct_model_opt = mjp.copy_ct_model(ct_model, optimal_angles)
    sinogram_opt = full_sino[optimal_angle_inds]
    recon_opt, recon_params = ct_model_opt.recon(sinogram_opt)

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
    ct_model_uniform = mjp.copy_ct_model(ct_model, uniform_angles)
    sino_uniform = full_sino[uniform_index_list]
    recon_uniform, recon_params_uniform = ct_model_uniform.recon(sino_uniform, max_iterations=max_iterations)

    mj.slice_viewer(recon_uniform, recon_opt, slice_label=['Uniform: Slice', 'VCLS optimal: Slice'],
                    title='Recons from {} views: \nuniformly spaced angles (left) and optimal angles (right)'.format(num_selected_views), vmin=0.0, vmax=0.05)

