"""
Hyperspectral Neutron Tomography
--------------------------------

Step 3 for the Ni-Cu-Al FHR demo: import the dehydrated sinogram from
output/hsnt/sino, reconstruct it, and export the dehydrated reconstruction to
output/hsnt/recon as an HDF5 file.
"""

import os
import numpy as np
import mbirjax as mj
import hsnt_prep_utils as h_preproc


# Setup paths
output_folder = 'output'
hsnt_output_folder = os.path.join(output_folder, 'hsnt')
sino_folder = os.path.join(hsnt_output_folder, 'sino')
recon_folder = os.path.join(hsnt_output_folder, 'recon')
input_file_name = os.path.join(sino_folder, 'dehydrated_sino.h5')
output_file_name = os.path.join(recon_folder, 'dehydrated_recon.h5')

# Setup parameters
alignment_offsets = [2, 2]  # Chip alignment offset values along the Y and X axes
center_offset = -0.25  # Center of rotation offset
recon_snr_db = 30  # Assumed SNR for the dataset in dB
verbose = 0  # Print nothing if 0

# Fix seed for random number generation
np.random.seed(129)


def main():
    print("---------------------------")
    print("STEP-3: MBIR RECONSTRUCTION")
    print("---------------------------")

    if not os.path.exists(input_file_name):
        raise FileNotFoundError(
            "Missing dehydrated sinogram file. Run demo_Ni_Cu_Al_FHR_step2.py first. "
            f"Expected file: {input_file_name}"
        )

    os.makedirs(recon_folder, exist_ok=True)

    hsnt_dehydrated_sino, metadata = mj.hsnt.import_hsnt_data_hdf5(input_file_name)
    subspace_data_all_angles, subspace_basis, dataset_type = hsnt_dehydrated_sino
    dataset_name = metadata.get('dataset_name', 'Unknown')
    print("Loaded dataset: ", metadata['dataset_name'])
    if metadata['angles'] is None:
        raise ValueError("Missing angles metadata in dehydrated sinogram file.")
    angles = metadata['angles']

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
