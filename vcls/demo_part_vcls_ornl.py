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

    ##############################################
    # Sets user selectable parameters
    ##############################################

    # ##### params for dataset downloading. User may change these parameters for their own datasets.
    # An example dataset will be downloaded from `dataset_url`, and saved to `download_dir`.
    # url to dataset.
    dataset_url_scan = '/depot/bouman/data/ORNL/hfn_scan.tgz'
    dataset_url_reference = '/depot/bouman/data/ORNL/hfn_reference_object.tgz'
    # destination path to download and extract the data and metadata.
    download_dir = './demo_data/'
    # Path to scan directory.
    dataset_dir_scan = mj.download_and_extract_tar(dataset_url_scan, download_dir)
    dataset_dir_reference = mj.download_and_extract_tar(dataset_url_reference, download_dir)

    # Load reference object
    print('Loading reference object')
    reference_object = np.load(os.path.join(dataset_dir_reference, f'reference_object.npy'))
    print('Done')

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

    #####################
    # Construct model
    #####################
    # Load and preprocess ORNL data
    # List all files ending in .h5 or .hdf5
    hdf5_files = sorted(
        f for f in os.listdir(dataset_dir_scan)
        if f.lower().endswith(('.h5', '.hdf5'))
    )
    filename = os.path.join(dataset_dir_scan, hdf5_files[0])
    full_sinogram, cone_beam_params_for_recon, optional_params_for_recon = out.compute_sino_and_params(filename)
    angle_candidates = cone_beam_params_for_recon['angles']  # This is probably not the best way to do this

    # Construct different params used in VCLS
    cone_beam_params_for_vcls = cone_beam_params_for_recon.copy()
    optional_params_for_vcls = optional_params_for_recon.copy()
    num_views = len(angle_candidates)
    num_det_rows = reference_object.shape[2]
    num_det_channels = reference_object.shape[0]
    sinogram_shape_for_vcls = (num_views, num_det_rows, num_det_channels)
    cone_beam_params_for_vcls["sinogram_shape"] = sinogram_shape_for_vcls

    # Construct cone beam object using ORNL parameters
    ct_model_for_vcls = mj.ConeBeamModel(**cone_beam_params_for_vcls)

    # Set optional ORNL geometry parameters
    ct_model_for_vcls.set_params(**optional_params_for_vcls)

    ##############################################
    # Run VCLS to Select Views and Display Results
    ##############################################
    time0 = time.time()
    optimal_angle_inds, vcl_value = mjp.get_opt_views(ct_model_for_vcls, reference_object, num_selected_views, r_1=r_1, r_2=r_2, verbose=1, seed=seed)
    optimal_angles = angle_candidates[optimal_angle_inds]
    elapsed = time.time() - time0
    print('Elapsed time for selected views is {:.3f} seconds'.format(elapsed))
    print('VCL value for selected views: {:.6f}'.format(vcl_value))

    # Display reference object cross-section with selected angles
    formatted = np.array2string(optimal_angles, precision=3, suppress_small=True, separator=', ')
    print('chosen angles: ' + formatted)
    mjp.show_image_with_projection_rays(reference_object[:, :, 0], rotation_angles_rad=optimal_angles, title='Reference Object with Selected View Angles')

    # Display reference object Fourier transform along with selected angles
    center_slice = reference_object[:, :, reference_object.shape[2] // 2]
    ref_fft = np.fft.fftshift(np.fft.fft2(center_slice))
    angles_perp = optimal_angles + np.pi / 2    # Add 90deg because Fourier transform of edge is perpendicular to edge
    mjp.show_image_with_projection_rays(np.log10(1e-2 + np.abs(ref_fft)), rotation_angles_rad=angles_perp, title='FFT of Reference Object\n with Selected View Angles')


    # Construct cone beam object for recon using ORNL parameters
    ct_model_for_recon = mj.ConeBeamModel(**cone_beam_params_for_recon)
    # Set optional ORNL geometry parameters
    ct_model_for_recon.set_params(**optional_params_for_recon)
    # Set recon parameters
    ct_model_for_recon.set_params(sharpness=sharpness, snr_db=snr_db)

    # Do a recon with optimal angles
    optimal_angles = angle_candidates[optimal_angle_inds]
    ct_model_opt = mjp.copy_ct_model(ct_model_for_recon, optimal_angles)
    sinogram_optimal_angles = full_sinogram[optimal_angle_inds]
    recon_optimal_angles, recon_params = ct_model_opt.recon(sinogram_optimal_angles, max_iterations=max_iterations)

    # Do a recon with uniform angles
    # Find uniform sampled index list
    num_det_channels_for_recon = cone_beam_params_for_recon["sinogram_shape"][2]
    source_detector_dist = cone_beam_params_for_recon["source_detector_dist"]
    detector_cone_angle = 2 * np.arctan2(num_det_channels_for_recon / 2, source_detector_dist)
    candidates_normalized = np.abs(angle_candidates - angle_candidates[0])
    end_index = np.where(candidates_normalized < np.pi + detector_cone_angle)[0][-1] # final angle in the short-scan range
    uniform_index_list = dut.create_uniform_index(angle_candidates, end_index, num_selected_views)
    uniform_angles = angle_candidates[uniform_index_list]

    ct_model_uniform = mjp.copy_ct_model(ct_model_for_recon, uniform_angles)
    sinogram_uniform = full_sinogram[uniform_index_list]
    recon_uniform, recon_params_uniform = ct_model_uniform.recon(sinogram_uniform, max_iterations=max_iterations)

    mj.slice_viewer(recon_uniform, recon_optimal_angles, slice_label=['Uniform: Slice', 'VCLS optimal: Slice'],
                    title='Recons from {} views: \nuniformly spaced angles (left) and optimal angles (right)'.format(num_selected_views), vmin=0.0, vmax=0.05)

