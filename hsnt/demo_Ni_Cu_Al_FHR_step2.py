"""
Hyperspectral Neutron Tomography
--------------------------------

Step 2 for the Ni-Cu-Al FHR demo: load preprocessed per-angle arrays from
output/hsnt/prep, dehydrate the sinogram, and export the dehydrated sinogram to
output/hsnt/sino as an HDF5 file.
"""

import os
import numpy as np
import mbirjax as mj


# Setup paths
dataset_name = 'Ni_Cu_Al_dataset'
output_folder = 'output'
hsnt_output_folder = os.path.join(output_folder, 'hsnt')
prep_folder = os.path.join(hsnt_output_folder, 'prep')
sino_folder = os.path.join(hsnt_output_folder, 'sino')
output_file_name = os.path.join(sino_folder, 'dehydrated_sino.h5')

# Setup parameters
num_materials = 3  # Number of materials in the sample
verbose = 0  # Print nothing if 0

# Fix seed for random number generation
np.random.seed(129)


def _processed_data_path(angle):
    return os.path.join(prep_folder, 'processed_data_' + str(angle) + '.npy')


def _load_prep_metadata():
    prep_metadata_file_name = os.path.join(prep_folder, 'metadata.npz')
    if not os.path.exists(prep_metadata_file_name):
        raise FileNotFoundError(
            "Missing prep metadata file. Run demo_Ni_Cu_Al_FHR_step1.py first. "
            f"Expected file: {prep_metadata_file_name}"
        )
    with np.load(prep_metadata_file_name) as prep_metadata:
        return prep_metadata['angles']


def main():
    print("------------------------------")
    print("STEP-2: LARGE DATA DEHYDRATION")
    print("------------------------------")

    os.makedirs(sino_folder, exist_ok=True)

    angles = _load_prep_metadata()
    missing_files = [path for path in (_processed_data_path(angle) for angle in angles) if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(
            "Missing preprocessed files. Run demo_Ni_Cu_Al_FHR_step1.py first. "
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

    # Pack dehydrated sinogram and save
    hsnt_dehydrated_sino = [subspace_data_all_angles, subspace_basis, dataset_type]
    metadata = mj.hsnt.create_hsnt_metadata(dataset_name=dataset_name, angles=np.array(angles))
    mj.hsnt.export_hsnt_data_hdf5(output_file_name, hsnt_dehydrated_sino, metadata)
    print("Saved dehydrated sinogram to: ", output_file_name)


if __name__ == '__main__':
    main()
