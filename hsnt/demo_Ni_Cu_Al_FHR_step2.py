"""
Hyperspectral Neutron Tomography
--------------------------------

Step 2 for the Ni-Cu-Al FHR demo: load preprocessed per-angle arrays from the
tmp folder, run dehydration and MBIR reconstruction, and export the dehydrated
reconstruction to an HDF5 file.
"""

import os
import numpy as np
import mbirjax as mj
import hsnt_prep_utils as h_preproc


# Setup paths
dataset_name = 'Ni_Cu_Al_dataset'
tmp_folder = 'tmp'
output_folder = 'output'
output_file_name = os.path.join(output_folder, 'dehydrated_recons_' + dataset_name + '.h5')

# Setup parameters
angles = [0.0, 6.2, 12.399, 16.231, 22.43, 32.461, 38.661, 42.492, 48.692, 58.723,
          64.922, 74.953, 81.153, 84.984, 91.184, 101.215, 107.415, 111.246, 117.446,
          127.477, 133.676, 143.707, 149.907, 153.738, 159.938, 169.969, 176.168]
num_materials = 3  # Number of materials in the sample
alignment_offsets = [2, 2]  # Chip alignment offset values along the Y and X axes
center_offset = -0.25  # Center of rotation offset
recon_snr_db = 30  # Assumed SNR for the dataset in dB
verbose = 0  # Print nothing if 0

# Fix seed for random number generation
np.random.seed(129)


def _processed_data_path(angle):
    return os.path.join(tmp_folder, 'processed_data_' + str(angle) + '.npy')


def main():
    print("------------------------------")
    print("STEP-2: LARGE DATA DEHYDRATION")
    print("------------------------------")

    os.makedirs(output_folder, exist_ok=True)

    missing_files = [path for path in (_processed_data_path(angle) for angle in angles) if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(
            "Missing preprocessed files. Run demo_Ni_Cu_Al_FHR_a_step1_preprocess.py first. "
            f"First missing file: {missing_files[0]}"
        )

    # Perform initial dehydration to estimate subspace basis vectors for each angle
    subspace_basis_all_angles = []
    for angle in angles:
        print("Currently estimating subspace basis for angle: ", angle)
        processed_data = np.load(_processed_data_path(angle))
        _, subspace_basis, _ = mj.hsnt.dehydrate(processed_data, num_materials=num_materials, verbose=verbose)
        subspace_basis_all_angles.append(subspace_basis)
    subspace_basis_all_angles = np.concatenate(subspace_basis_all_angles, axis=0)

    # Estimate refined set of subspace basis vectors combining estimations for all angles
    print("Refining subspace basis")
    _, subspace_basis, dataset_type = mj.hsnt.dehydrate(subspace_basis_all_angles, num_materials=num_materials,
                                                        verbose=verbose)

    # Perform final dehydration for each angle using the refined subspace basis vectors
    subspace_data_all_angles = []
    for angle in angles:
        print("Currently estimating subspace data for angle: ", angle)
        processed_data = np.load(_processed_data_path(angle))
        subspace_data, _, _ = mj.hsnt.dehydrate(processed_data, subspace_basis=subspace_basis, verbose=verbose)
        subspace_data_all_angles.append(subspace_data)
    subspace_data_all_angles = np.concatenate(subspace_data_all_angles, axis=0)

    print("---------------------------")
    print("STEP-3: MBIR RECONSTRUCTION")
    print("---------------------------")

    # Fix the chip alignment issues for proper reconstruction
    subspace_data_all_angles = h_preproc.correct_alignment_ORNL_SNAP(subspace_data_all_angles, alignment_offsets)

    # MBIR model setup
    angles_r = np.array(angles) * np.pi / 180  # Convert the angles to radian
    num_angles, detector_rows, detector_columns, subspace_dimension = subspace_data_all_angles.shape
    mj_model = mj.ParallelBeamModel((num_angles, detector_rows, detector_columns), angles_r)
    mj_model.set_params(snr_db=recon_snr_db, sharpness=0, det_channel_offset=center_offset, verbose=verbose)

    # Perform MBIR
    subspace_recons = []
    for idx in range(subspace_dimension):
        print("Reconstructing data for subspace index: " + str(idx))
        subspace_recon, _ = mj_model.recon(subspace_data_all_angles[:, :, :, idx])
        subspace_recons.append(subspace_recon)
    subspace_recons = np.moveaxis(np.array(subspace_recons), 0, -1)

    # Pack dehydrated reconstructions and save
    hsnt_dehydrated_recons = [subspace_recons, subspace_basis, dataset_type]
    metadata = mj.hsnt.create_hsnt_metadata(dataset_name=dataset_name)
    mj.hsnt.export_hsnt_data_hdf5(output_file_name, hsnt_dehydrated_recons, metadata)
    print("Saved dehydrated reconstruction to: ", output_file_name)


if __name__ == '__main__':
    main()
