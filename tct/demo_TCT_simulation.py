import sys
import importlib.util
import subprocess

# Install necessary package
required_packages = ['trimesh']
for package in required_packages:
    if importlib.util.find_spec(package) is None:
        print(f"{package} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}. Please install it manually.")
            sys.exit(1)
    else:
        print(f"{package} is already installed.")

import mbirjax as mj
import mbirjax.preprocess as mjp
import numpy as np
import os
import trimesh
from scipy.ndimage import zoom

def get_experiment_params(experiment_name):
    """Returns experiment-specific parameters."""
    base_config = {
        "source_det_dist_mm": 120,
        "source_iso_dist_mm": 100,
        "det_pixel_pitch_mm": 75 / 1000,
        "x_view_space_mm": 14,
        "z_view_space_mm": 7,
        "num_x_translations": 11,
        "num_z_translations": 9,
        "num_det_rows": 1936,
        "num_det_channels": 3064,
        "num_recon_rows": 72,
        "qggmrf_nbr_weights": [0.1, 1.0, 1.0],
        "sharpness": 1,
    }

    if experiment_name == "experiment1":
        return base_config

    elif experiment_name == "experiment2":
        override_config = {
            # Modify only what's different for experiment2
            # Example: "x_view_space_mm": 5,
        }
        return {**base_config, **override_config}

    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")


if __name__ == "__main__":
    print("This script runs TCT simulation experiment using a CAD model of circuit board")

    experiment = sys.argv[1] if len(sys.argv) > 1 else "experiment1"
    params = get_experiment_params(experiment)

    # Download and extract CAD file
    download_dir = "./demo_data/"
    dataset_url = "https://www.datadepot.rcac.purdue.edu/bouman/elec_board/Elec_Board_CAD_and_License.tgz"
    cad_file_dir = mj.download_and_extract(dataset_url, download_dir)
    cad_file_path = os.path.join(download_dir, "Elec_Board_CAD_and_License/5723 Feather RP2040 USB Host/5723 Feather RP2040 USB Host.stl")

    # Output path
    output_path = './output/'  # path to store ground truth phantom and synthetic sinogram

    # Set parameters for experiment
    source_det_dist_mm = params["source_det_dist_mm"]
    source_iso_dist_mm = params["source_iso_dist_mm"]
    det_pixel_pitch_mm = params["det_pixel_pitch_mm"]
    x_view_space_mm = params["x_view_space_mm"]
    z_view_space_mm = params["z_view_space_mm"]
    num_x_translations = params["num_x_translations"]
    num_z_translations = params["num_z_translations"]
    num_det_rows = params["num_det_rows"]
    num_det_channels = params["num_det_channels"]
    num_recon_rows = params["num_recon_rows"]
    qggmrf_nbr_weights = params["qggmrf_nbr_weights"]
    sharpness = params["sharpness"]

    # Calculate physical parameters in ALU
    # Note: 1 ALU = 1 delta_det_channel_unit
    # Unit conversion table
    unit_conversion = {'um': 1.0, 'mm': 1000.0}

    # Set 1 ALU = 1 delta_det_channel_unit
    ALU_unit = 'mm'
    ALU_value = 1

    # Convert physical units to ALU
    print("\n********** Convert parameters from physical units to ALU units **************")
    delta_det_channel_ALU = det_pixel_pitch_mm * unit_conversion['mm'] / unit_conversion[ALU_unit]
    delta_det_row_ALU = delta_det_channel_ALU
    source_iso_dist_ALU = source_iso_dist_mm * unit_conversion['mm'] / unit_conversion[ALU_unit]
    source_det_dist_ALU = source_det_dist_mm * unit_conversion['mm'] / unit_conversion[ALU_unit]
    x_view_spacing_ALU = x_view_space_mm * unit_conversion['mm'] / unit_conversion[ALU_unit]
    z_view_spacing_ALU = z_view_space_mm * unit_conversion['mm'] / unit_conversion[ALU_unit]
    half_angle_rad = np.arctan2(max(num_det_rows, num_det_channels) / 2.0, source_iso_dist_ALU)

    # Generate translation vectors
    print("\n********** Generate translation vectors **************")
    translation_vectors = mj.gen_translation_vectors(num_x_translations, num_z_translations, x_view_spacing_ALU, z_view_spacing_ALU)

    # Compute sinogram shape
    sino_shape = (translation_vectors.shape[0], num_det_rows, num_det_channels)

    # Initialize model for forward projection
    print("\n********** Construct translation model for forward projection **************")
    tct_model = mj.TranslationModel(sino_shape, translation_vectors, source_detector_dist=source_det_dist_ALU, source_iso_dist=source_iso_dist_ALU)

    # Calculate recon_shape, delta_voxel, and delta_recon_row parameters
    recon_shape, delta_voxel, delta_recon_row = mj.utilities.calc_tct_recon_params(source_det_dist_ALU,
                                                                     source_iso_dist_ALU,
                                                                     delta_det_row_ALU,
                                                                     delta_det_channel_ALU, sino_shape,
                                                                     translation_vectors)

    ### Generate ground truth phantom
    print("\n********** Generate ground truth phantom **************")
    # Load the CAD file and turn it into a 3D phantom (object=1, background=0)
    gt_phantom = trimesh.load(cad_file_path).voxelized(pitch=0.2).matrix.astype(np.uint8)

    # Resize the phantom to match desired resolution
    gt_phantom = zoom(gt_phantom, zoom=(0.2 / delta_voxel, 0.2 / delta_voxel, 0.2 / delta_recon_row), order=0)
    gt_phantom = gt_phantom.transpose(2, 1, 0)

    # Define recon shape for simulated phantom
    recon_shape = (int(2.0 * gt_phantom.shape[0]), recon_shape[1], int(recon_shape[2]*1.1))

    # Reshape simulated phantom to recon shape
    pad_total = np.array(recon_shape) - np.array(gt_phantom.shape)
    pads = [(pad_total[i] // 2, pad_total[i] - pad_total[i] // 2) for i in range(3)]
    gt_phantom = np.pad(gt_phantom, pads, mode='constant')

    # Set model parameters
    tct_model.set_params(positivity_flag=True)
    tct_model.set_params(partition_sequence=5*[0,] + 100*[1, 3,])
    tct_model.set_params(delta_det_channel=delta_det_channel_ALU)
    tct_model.set_params(delta_det_row=delta_det_row_ALU)
    tct_model.set_params(delta_voxel=delta_voxel)
    tct_model.set_params(recon_shape=recon_shape)
    tct_model.set_params(delta_recon_row=delta_recon_row)
    tct_model.set_params(qggmrf_nbr_wts=qggmrf_nbr_weights)
    tct_model.set_params(sharpness=sharpness)
    tct_model.set_params(alu_unit=ALU_unit)
    tct_model.set_params(alu_value=ALU_value)

    # Display translation array
    translation_vectors_display = np.asarray(translation_vectors).copy()
    translation_vectors_display[:, 0] /= delta_voxel
    translation_vectors_display[:, 2] /= delta_voxel
    translation_vectors_display[:, 1] /= delta_recon_row
    mj.display_translation_vectors(translation_vectors_display, recon_shape)

    # View ground truth phantom
    mj.slice_viewer(gt_phantom.transpose(0, 2, 1), title='Ground Truth Recon', slice_label='View', slice_axis=0)

    # Generate synthetic sonogram data
    print("\n********** Generate forward projections of the phantom **************")
    sino = tct_model.forward_project(gt_phantom)
    sino = np.asarray(sino)

    # View synthetic sinogram
    mj.slice_viewer(sino, slice_axis=0, title='Synthetic sinogram', slice_label='View')

    # Set reconstruction shape
    tct_model.set_params(recon_shape=(num_recon_rows,)+recon_shape[1:])

    # Print out model parameters
    tct_model.print_params()

    # Perform MBIR reconstruction
    print("\n********** Perform MBIR reconstruction **************")
    mbir_recon, mbir_dict = tct_model.recon(sino, max_iterations=200, stop_threshold_change_pct=0.3)

    # Save reconstruction results
    print("\n*********** save MBIR recon in h5 format *************")
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, f'TCT_simulation_recon.h5')
    mj.export_recon_hdf5(output_path, mbir_recon, recon_dict=mbir_dict, top_margin=0, bottom_margin=0)

    # Display results
    mj.slice_viewer(gt_phantom.transpose(0, 2, 1), mbir_recon.transpose(0, 2, 1),
                    vmin=0, vmax=0.8, title='Object (left), MBIR reconstruction (right)',
                    slice_axis=0)