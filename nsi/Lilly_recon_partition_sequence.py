import numpy as np
import os
import pprint
import argparse
import mbirjax as mj
import mbirjax.preprocess as mjp

pp = pprint.PrettyPrinter(indent=4)

# === Predefined partition sequences ===
PARTITION_SEQUENCES = {
    "default":  [0, 2, 4, 6, 7],          # mbirjax default (includes granularity 1)
    "skip_0":    [2, 4, 6, 7],             # 4,16,64,128
}

if __name__ == "__main__":
    # === MBIR Recon parameters ===
    sharpness = 1.0
    verbose = 1

    # ----------------------------
    # Parse command line arguments
    # ----------------------------
    parser = argparse.ArgumentParser(description="MBIRJAX Plastic-Metal Reconstruction Demo")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to existing data directory.")
    parser.add_argument("--downsampling", type=int, default=1,  # Perhaps change to subsample_detector_factor
                        help="Subsampling factor for detector rows and channels.")
    parser.add_argument("--subsample_view_factor", type=int, default=1,
                        help="Subsampling factor for projection views.")
    parser.add_argument("--num_metal", type=int, default=2,
                        help="Number of metal types for segmentation and MAR.")
    parser.add_argument("--sino_cropping", type=int, default=1,
                        help="Flag for applying sinogram cropping")
    parser.add_argument("--partition_sequence", type=str, default="default",
                        choices=list(PARTITION_SEQUENCES.keys()),
                        help="Name of a predefined partition sequence (see PARTITION_SEQUENCES).")
    parser.add_argument("--max_iterations", type=int, default=15,
                        help="Maximum number of MBIR iterations.")
    args = parser.parse_args()

    # Set output path
    output_path = './output/lilly/'   # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

    # Set log path
    logfile_path = './logs/'   # path to store output logs
    os.makedirs(logfile_path, exist_ok=True)  # mkdir if directory does not exist

    if args.data_path is not None and not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"--data_path does not exist or is not a directory: {args.data_path}")

    # Get parameters from command line
    dataset_dir = args.data_path
    downsample = args.downsampling
    num_metal = args.num_metal
    subsample_view_factor = args.subsample_view_factor
    partition_sequence_name = args.partition_sequence
    partition_sequence = PARTITION_SEQUENCES[partition_sequence_name]
    max_iterations = args.max_iterations

    # Set program parameters
    downsample_rate = [downsample, downsample]
    dataset_tag = os.path.basename(dataset_dir.rstrip("/"))

    if verbose>0:
        print("\n************** NSI dataset preprocessing **************")
    sino, cone_beam_params, optional_params = \
        mjp.nsi.compute_sino_and_params(dataset_dir, downsample_factor=downsample_rate, subsample_view_factor=subsample_view_factor)

    cropping = bool(args.sino_cropping)
    if cropping:
        if verbose>0:
            print("\n********** Cropping sinogram margins and update cone-beam geometry parameters **********")
        sino, cone_beam_params, optional_params = mjp.auto_crop_sino_conebeam(sino, cone_beam_params, optional_params)

    # Using np.maximum (not jnp.maximum) keeps the full sinogram in host memory.
    sino = np.maximum(sino, 0.0)

    if verbose>0:
        print("\n***************** Set up MBIRJAX model ****************")
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    ct_model.set_params(**optional_params)
    ct_model.set_params(sharpness=sharpness, verbose=verbose, positivity_flag=True)
    ct_model.set_params(partition_sequence=partition_sequence)
    if verbose>0:
        print(f"Using partition sequence '{partition_sequence_name}': {partition_sequence}")
    weights_trans = mj.gen_weights(sino, weight_type='transmission_root')
    if verbose>0:
        ct_model.print_params()

    # Set location of log file
    logfile_path = os.path.expanduser(f"{logfile_path}recon_{dataset_tag}_pseq_{partition_sequence_name}.log")

    if verbose>0:
        print("\n*************** Compute reconstruction ***************")
    if num_metal == 0:
        # perform standard MBIR recon using split sino to reduce memory
        recon, _ = ct_model.split_sino_recon(sino, weights=weights_trans, max_iterations=max_iterations,
                                             logfile_path=logfile_path)
    else:
        # perform MAR recon
        recon = mjp.recon_plastic_metal(ct_model, sino, weights_trans, num_metal=num_metal, verbose=verbose)
    if verbose>0 and num_metal == 0:
        print(f"Saved mbirjax recon log to {logfile_path}")

    # Load voxel pitch
    delta_voxel_mm = ct_model.get_params('delta_voxel') * ct_model.get_params('alu_value')
    delta_voxel_um = delta_voxel_mm * 1000
    # Save recon to hdf5
    if verbose>0:
        print("\n*********** save mar and fdk recon in h5 format *************")
    hdf5_path = os.path.join(output_path,
    f"recon_{dataset_tag}_nummetal_{num_metal}_voxel_pitch_{delta_voxel_um:.2f}um_pseq_{partition_sequence_name}.h5")
    mj.export_recon_hdf5(hdf5_path, recon, recon_dict=None, remove_flash=True)
    if verbose>0:
        print("Metal artifact reduction recon saved to {}".format(os.path.abspath(hdf5_path)))
