seed = 42 # Change this value to control randomness across runs

import time
import jax.numpy as jnp
import mbirjax as mj
import numpy as np
import utils as dut
import vcls


if __name__ == '__main__':
    
    ##############################################
    # Set user-selectable parameters
    ##############################################
    geometry_type = 'cone'  # 'cone' or 'parallel'
    num_object_rows = 128
    num_object_slices = 64
    magnification = 2.0
    cone_angle = (15/180)*np.pi
    num_candidate_views = 128
    #num_selected_views = 25
    num_selected_views = 15
    start_angle = 0
    end_angle = 2 * np.pi

    # Voxel sampling rate within the ROI.
    r_1 = 0.5
    r_2 = 1.0

    ##############################################
    # Create the reference object and 3D ROI
    ##############################################
    print('Creating phantom and artificial ROI')
    reference_object = dut.gen_polygon_phantom(
        num_rows=num_object_rows, num_slices=num_object_slices
    )
    roi = dut.create_artificial_roi(reference_object)
    print(f'reference_object shape: {reference_object.shape}')
    print(f'ROI voxels: {np.count_nonzero(roi)}')
    dut.show_roi(reference_object, roi)

    ##############################################
    # Construct the CT model
    ##############################################
    num_det_rows = reference_object.shape[2]
    num_det_channels = reference_object.shape[0]
    sinogram_shape = (num_candidate_views, num_det_rows, num_det_channels)

    source_detector_dist = (
        1.0 / np.tan(cone_angle / 2.0)
    ) * (num_det_channels / 2)
    source_iso_dist = source_detector_dist / magnification
    angle_candidates = jnp.linspace(
        start_angle, end_angle, num_candidate_views, endpoint=False
    )
    ct_model = mj.get_ct_model(
        geometry_type, sinogram_shape, angle_candidates,
        source_detector_dist, source_iso_dist
    )
    ct_model.set_params(recon_shape=reference_object.shape)

    ##############################################
    # Select views without and with ROI voxels
    ##############################################
    start_time = time.time()
    no_roi_angle_inds, no_roi_vcl_value = vcls.get_opt_views(
        ct_model,
        reference_object,
        num_selected_views,
        r_1=r_1,
        r_2=r_2,
        verbose=0,
        seed=seed,
    )
    no_roi_angles = angle_candidates[no_roi_angle_inds]
    no_roi_elapsed = time.time() - start_time

    start_time = time.time()
    roi_angle_inds, roi_vcl_value = vcls.get_opt_views(
        ct_model,
        reference_object,
        num_selected_views,
        r_1=r_1,
        r_2=r_2,
        verbose=0,
        seed=seed,
        roi=roi,
    )
    roi_angles = angle_candidates[roi_angle_inds]
    roi_elapsed = time.time() - start_time

    uniform_angles = jnp.linspace(
        start_angle, end_angle, num_selected_views, endpoint=False
    )

    print('EVCL without ROI elapsed time: {:.3f} seconds'.format(no_roi_elapsed))
    print('EVCL without ROI value: {:.6f}'.format(no_roi_vcl_value))
    print('EVCL without ROI view indices:', no_roi_angle_inds)
    print(
        'EVCL without ROI angles:',
        np.array2string(
            np.asarray(no_roi_angles), precision=3,
            suppress_small=True, separator=', '
        ),
    )

    print('EVCL with ROI elapsed time: {:.3f} seconds'.format(roi_elapsed))
    print('EVCL with ROI value: {:.6f}'.format(roi_vcl_value))
    print('EVCL with ROI view indices:', roi_angle_inds)
    print(
        'EVCL with ROI angles:',
        np.array2string(
            np.asarray(roi_angles), precision=3,
            suppress_small=True, separator=', '
        ),
    )

    dut.show_view_angle_comparison(
        reference_object[:, :, 0],
        [uniform_angles, no_roi_angles, roi_angles],
        ['Uniform Angles', 'EVCL without ROI', 'EVCL with ROI'],
    )

    ##############################################
    # Compare uniform, no-ROI, and ROI recons
    ##############################################
    ct_model_uniform = mj.copy_ct_model(ct_model, uniform_angles)
    ct_model_uniform.set_params(recon_shape=reference_object.shape)
    sinogram_uniform = ct_model_uniform.forward_project(reference_object)
    recon_uniform, recon_dict_uniform = ct_model_uniform.recon(sinogram_uniform)

    ct_model_no_roi = mj.copy_ct_model(ct_model, no_roi_angles)
    ct_model_no_roi.set_params(recon_shape=reference_object.shape)
    sinogram_no_roi = ct_model_no_roi.forward_project(reference_object)
    recon_no_roi, recon_dict_no_roi = ct_model_no_roi.recon(sinogram_no_roi)

    ct_model_roi = mj.copy_ct_model(ct_model, roi_angles)
    ct_model_roi.set_params(recon_shape=reference_object.shape)
    sinogram_roi = ct_model_roi.forward_project(reference_object)
    recon_roi, recon_dict_roi = ct_model_roi.recon(sinogram_roi)

    mj.slice_viewer(
        reference_object,
        recon_uniform,
        recon_no_roi,
        recon_roi,
        data_dicts=[None, recon_dict_uniform, recon_dict_no_roi, recon_dict_roi],
        slice_label=[
            'Reference Object', 'Uniform Angles',
            'EVCL without ROI', 'EVCL with ROI'
        ],
        title=(
            'Reference object plus reconstructions from uniform,\n'
            'EVCL without ROI, and EVCL with ROI angles'
        ),
    )
