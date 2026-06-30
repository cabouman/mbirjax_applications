import numpy as np
import os
import pprint
import argparse
import mbirjax as mj
import mbirjax.preprocess as mjp

pp = pprint.PrettyPrinter(indent=4)

# === Predefined partition sequences ===
# Values are INDICES into the default granularity [1, 2, 4, 8, 16, 32, 64, 128, 256],
# so index 2->granularity 4, ..., index 7->granularity 128.  Starting at index 2 keeps
# granularity >= 4 (avoids the memory-heavy granularity-1 step on very large recons),
# and the coarsest used is index 7 (granularity 128).
PARTITION_SEQUENCES = {
    "default":      [0, 2, 4, 6, 7],          # mbirjax default (includes granularity 1)
    "coarse_4_128": [2, 3, 4, 5, 6, 7],       # 4,8,16,32,64,128
    "slow_start":   [2, 2, 3, 4, 5, 6, 7],    # linger at granularity 4 before progressing
    "slow_dip":     [2, 3, 2, 4, 5, 6, 7],    # 4,8,4,16,32,64,128
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
    parser.add_argument("--partition_sequence", type=str, default="coarse_4_128",
                        choices=list(PARTITION_SEQUENCES.keys()),
                        help="Name of a predefined partition sequence (see PARTITION_SEQUENCES).")
    args = parser.parse_args()

    # Set output path
    output_path = './output/lilly/'   # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

    if args.data_path is not None and not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"--data_path does not exist or is not a directory: {args.data_path}")

    # Get parameters from command line
    dataset_dir = args.data_path
    downsample = args.downsampling
    num_metal = args.num_metal
    subsample_view_factor = args.subsample_view_factor
    partition_sequence_name = args.partition_sequence
    partition_sequence = PARTITION_SEQUENCES[partition_sequence_name]

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

    # Clip sinogram to be non-negative, ON THE HOST.  Using np.maximum (not jnp.maximum) keeps the full
    # sinogram in host memory: jnp.maximum would copy the whole sinogram onto one GPU (and gen_weights
    # below would add a second full-sino GPU array), only for split_sino_recon to gather them back to the
    # host -- a wasteful round-trip that can OOM a single GPU before the recon starts.  Kept host, each
    # half-recon shards its own half from the host.
    sino = np.maximum(sino, 0.0)

    if verbose>0:
        print("\n***************** Set up MBIRJAX model ****************")
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    ct_model.set_params(**optional_params)
    ct_model.set_params(sharpness=sharpness, verbose=verbose, positivity_flag=True)
    # Override the partition sequence to control recon granularity (memory vs. convergence tradeoff)
    ct_model.set_params(partition_sequence=partition_sequence)
    if verbose>0:
        print(f"Using partition sequence '{partition_sequence_name}': {partition_sequence}")
    weights_trans = mj.gen_weights(sino, weight_type='transmission_root')
    if verbose>0:
        ct_model.print_params()

    # Per-sequence mbirjax log path, so each partition-sequence run keeps its own log.
    logfile_path = os.path.expanduser(f"~/mbirjax_notes/recon_{dataset_tag}_pseq_{partition_sequence_name}.log")

    if verbose>0:
        print("\n*************** Compute reconstruction ***************")
    if num_metal == 0:
        # No metal artifact reduction: this is a plain MBIR recon, so call split_sino_recon
        # directly (recon_plastic_metal does the same internally for num_metal==0) and use its
        # native logfile_path option to write the detailed log into ~/mbirjax_notes.  We pass
        # stop_threshold_change_pct=0.5 to match recon_plastic_metal's default for apples-to-apples.
        recon, _ = ct_model.split_sino_recon(sino, weights=weights_trans, stop_threshold_change_pct=0.5,
                                             logfile_path=logfile_path)
    else:
        # recon_plastic_metal does not forward logfile_path, so this case logs to the mbirjax
        # default location (~/.mbirjax/logs/recon.log).
        recon = mjp.recon_plastic_metal(ct_model, sino, weights_trans, num_metal=num_metal, verbose=verbose)
    if verbose>0 and num_metal == 0:
        print(f"Saved mbirjax recon log to {logfile_path}")

    # Load voxel pitch
    delta_voxel_mm = ct_model.get_params('delta_voxel') * ct_model.get_params('alu_value')
    delta_voxel_um = delta_voxel_mm * 1000
    # Save recon to hdf5
    if verbose>0:
        print("\n*********** save mar and fdk recon in h5 format *************")
    mar_path = os.path.join(output_path,
    f"recon_{dataset_tag}_nummetal_{num_metal}_voxel_pitch_{delta_voxel_um:.2f}um_pseq_{partition_sequence_name}_mar.h5")
    mj.export_recon_hdf5(mar_path, recon, recon_dict=None, remove_flash=True)
    if verbose>0:
        print("Metal artifact reduction recon saved to {}".format(os.path.abspath(mar_path)))
