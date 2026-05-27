"""
Hyperspectral Neutron Tomography
--------------------------------

Step 4 for the Ni-Cu-Al FHR demo: import the dehydrated reconstruction from
output/hsnt/recon, partially rehydrate selected wavelength indices, and display
the result.
"""

import os
import mbirjax as mj
import plot_utils as p_utils


# Setup paths
dataset_name = 'Ni_Cu_Al_dataset'
output_folder = 'output'
hsnt_output_folder = os.path.join(output_folder, 'hsnt')
recon_folder = os.path.join(hsnt_output_folder, 'recon')
input_file_name = os.path.join(recon_folder, 'dehydrated_recons_' + dataset_name + '.h5')

# Display parameters
disp_wave_idx = [300, 600, 900]
disp_slices = [80, 200, 360]


def main():
    print("-------------------------------------------")
    print("STEP-4: PARTIAL REHYDRATION & VISUALIZATION")
    print("-------------------------------------------")

    if not os.path.exists(input_file_name):
        raise FileNotFoundError(
            "Missing dehydrated reconstruction file. Run demo_Ni_Cu_Al_FHR_step3.py first. "
            f"Expected file: {input_file_name}"
        )

    hsnt_dehydrated_recons, metadata = mj.import_hsnt_data_hdf5(input_file_name)
    print("Loaded dataset: ", metadata['dataset_name'])

    # Rehydrate only the display wavelength reconstruction
    hsnt_recon = mj.rehydrate(hsnt_dehydrated_recons, hyperspectral_idx=disp_wave_idx)

    # Plot image
    print("Displaying reconstructed image for wavelength indices: ", disp_wave_idx, ", and slice indices: ", disp_slices)
    rehydrated_idx = [i for i in range(len(disp_wave_idx))]
    p_utils.plot_hyper_recons(hsnt_recon, display_wave_idx=rehydrated_idx, display_slices=disp_slices)


if __name__ == '__main__':
    main()
