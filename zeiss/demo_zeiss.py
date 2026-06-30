import os

# Fraction of each GPU's memory JAX may preallocate (default 0.75 leaves ~25% idle).
# MUST be set before 'import jax'
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import sys
import numpy as np
import jax
import jax.numpy as jnp
import pprint
import mbirjax as mj
import mbirjax.preprocess as mjp

pp = pprint.PrettyPrinter(indent=4)

# === Partition-sequence experiments (granularity vs. GPU memory) ===
# Values are INDICES into the default granularity [1, 2, 4, 8, 16, 32, 64, 128, 256],
# granularity 1 (index 0) does a full-image preconditioned-gradient step, 
# which has the highest per-device peak memory; 
# starting coarser (index >= 2) reduces the memory demand.
PARTITION_SEQUENCES = {
    "default":      [0, 2, 4, 6, 7],          # mbirjax default (includes granularity 1)
    "coarse_4_128": [2, 3, 4, 5, 6, 7],       # 4,8,16,32,64,128
    "slow_start":   [2, 2, 3, 4, 5, 6, 7],    # linger at granularity 4 before progressing
    "slow_dip":     [2, 3, 2, 4, 5, 6, 7],    # 4,8,4,16,32,64,128
}
# Per-dataset defaults, used when a dataset entry below does not specify its own
# 'partition_sequence' / 'max_iterations' key.
DEFAULT_PARTITION_SEQUENCE_NAME = "default"   # a key into PARTITION_SEQUENCES above
DEFAULT_MAX_ITERATIONS = 15                    # mbirjax recon() default


def report_peak_gpu_memory(label=""):
    """Print the per-device peak GPU memory high-water mark.

    peak_bytes_in_use is cumulative since process start (not a snapshot), so for this script --
    which performs a single recon per invocation -- it reports the true peak the run required.
    The MAX over devices is the number that determines whether the recon fits on one GPU.
    """
    print(f"\n********** Peak GPU memory usage {label} **************")
    peak_per_device = []
    for d in jax.devices():
        try:
            peak = d.memory_stats().get('peak_bytes_in_use')
        except Exception:
            peak = None
        if peak is None:
            print(f"  {d}: peak_bytes_in_use unavailable (not a GPU?)")
            continue
        peak_per_device.append(peak)
        print(f"  {d}: peak {peak / 2**30:.2f} GiB")
    if peak_per_device:
        print(f"  Max over devices:  {max(peak_per_device) / 2**30:.2f} GiB  (the per-GPU fit constraint)")
        print(f"  Sum over devices:  {sum(peak_per_device) / 2**30:.2f} GiB")

if __name__ == "__main__":
    print("This script is for reconstructing cone beam CT data from Zeiss scanner")

    # Recon parameters
    verbose = 1               # Print, but do not display plots

    # Output path
    output_path = './output/zeiss_demo/'  # path to store output recon images

    # Define available datasets and parameters
    available_datasets = {
        'ORNL Z62': {
            'url': '/depot/bouman/data/ORNL/versa/ParAM-Round-1_Z62.txrm',
            'sharpness': 1.5,
            'snr_db': 35.0,
            'downsample_factor': 1,
            'subsample_view_factor': 2,
            'view_alignment': False,
            'vmin': 0,
            'vmax': 0.4,
            'partition_sequence': 'coarse_4_128',  # skip granularity 1 so 2k^3 fits in GPU memory
            'max_iterations': 30,                  # 2k^3 Z62 still changing >0.5%/iter at 15; adjust as needed
        },
        'ORNL SiC Composite': {
            'url': '/depot/bouman/data/ORNL/versa/SiC-SiC_CompositeFFOV_tomo-A.txrm',
            'sharpness': 1.5,
            'snr_db': 35.0,
            'downsample_factor': 2,
            'subsample_view_factor': 2,
            'view_alignment': True,
            'vmin': 0,
            'vmax': 0.4,
        },
        'Purdue BGA HART scan': {
            'url': '/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_HART_360_HART.txrm',
            'sharpness': 1.5,
            'snr_db': 35.0,
            'downsample_factor': 2,
            'subsample_view_factor': 2,
            'view_alignment': False,
            'vmin': 0,
            'vmax': 1,
        },
        'Purdue BGA Normal scan': {
            'url': '/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_No_HART.txrm',
            'sharpness': 1.5,
            'snr_db': 35.0,
            'downsample_factor': 2,
            'subsample_view_factor': 2,
            'view_alignment': False,
            'vmin': 0,
            'vmax': 1,
        },
        'Zeiss Synthetic Foam': {
            'url': '/depot/bouman/data/Zeiss/foam512R1N3000_raw_scan.txrm',
            'sharpness': 0.25,
            'snr_db': 30.0,
            'downsample_factor': 1,
            'subsample_view_factor': 1,
            'view_alignment': False,
            'vmin': 0,
            'vmax': 10,
        },
        'Purdue Sample': {
            'url': '/depot/bouman/data/Zeiss/purdue/Scan_tomo-A.txrm',
            'sharpness': 1.5,
            'snr_db': 35.0,
            'downsample_factor': 2,
            'subsample_view_factor': 2,
            'view_alignment': False,
            'vmin': 0,
            'vmax': 10,
        },
        'Nano CT Sample B': {
            'url': '/depot/bouman/data/AFRL/lipp/Black_Sheep_tomo-B_CS-2.txrm',
            'sharpness': 2.0,
            'snr_db': 35.0,
            'downsample_factor': 1,
            'subsample_view_factor': 1,
            'view_alignment': False,
            'vmin': 0,
            'vmax': 0.001,
        },
        'Nano CT Sample C': {
            'url': '/depot/bouman/data/AFRL/lipp/Black_Sheep_tomo-C_CS0.txrm',
            'sharpness': 2.0,
            'snr_db': 35.0,
            'downsample_factor': 1,
            'subsample_view_factor': 1,
            'view_alignment': False,
            'vmin': 0,
            'vmax': 0.001,
        },
        'Nano CT Sample D': {
            'url': '/depot/bouman/data/AFRL/lipp/Black_Sheep_tomo-D_CS-3.8.txrm',
            'sharpness': 2.0,
            'snr_db': 35.0,
            'downsample_factor': 1,
            'subsample_view_factor': 1,
            'view_alignment': False,
            'vmin': 0,
            'vmax': 0.001,
        },
    }

    # Prompt user for dataset selection using a numbered menu
    dataset_names = list(available_datasets.keys())
    print("Available datasets:")
    for i, name in enumerate(dataset_names, 1):
        print(f"{i}. {name}")
    try:
        selection = int(input("Enter the number of the dataset to reconstruct: "))
        if 1 <= selection <= len(dataset_names):
            dataset = dataset_names[selection - 1]
        else:
            raise ValueError
    except ValueError:
        print("Invalid selection.")
        sys.exit(1)

    # Set values of data set specific parameters
    dataset_url = available_datasets[dataset]['url']
    sharpness = available_datasets[dataset]['sharpness']
    snr_db = available_datasets[dataset]['snr_db']
    downsample_factor = available_datasets[dataset]['downsample_factor']
    subsample_view_factor = available_datasets[dataset]['subsample_view_factor']
    view_alignment = available_datasets[dataset]['view_alignment']
    vmin = available_datasets[dataset]['vmin']
    vmax = available_datasets[dataset]['vmax']
    # Optional per-dataset recon controls; fall back to defaults when not specified.
    partition_sequence_name = available_datasets[dataset].get('partition_sequence', DEFAULT_PARTITION_SEQUENCE_NAME)
    max_iterations = available_datasets[dataset].get('max_iterations', DEFAULT_MAX_ITERATIONS)

    # Load the sinogram and metadata
    print("\n********** Load sinogram and metadata from the data **************")
    sinogram, geometry_params, optional_params, zeiss_metadata = mjp.zeiss.compute_sino_and_params(dataset_url, downsample_factor=(downsample_factor, downsample_factor),
                                                                                                 subsample_view_factor=subsample_view_factor)

    # Construct tomography model
    print("\n********** Construct tomography model **************")
    if zeiss_metadata['scanner_type'] == 'ultra':
        ct_model = mj.ParallelBeamModel(**geometry_params)
        ct_model.set_params(**optional_params)
    else:
        ct_model = mj.ConeBeamModel(**geometry_params)
        ct_model.set_params(**optional_params)

    # Rerun auto-parameter functions because we changed the assumed detector pitch
    ct_model.auto_set_recon_geometry() # Reset default recon shape

    # Sharpness and snr_db
    ct_model.set_params(sharpness=sharpness, snr_db=snr_db, verbose=1)

    # Override the partition sequence to control recon granularity (memory vs. convergence tradeoff)
    partition_sequence = PARTITION_SEQUENCES[partition_sequence_name]
    ct_model.set_params(partition_sequence=partition_sequence)
    print(f"Using partition sequence '{partition_sequence_name}': {partition_sequence}")
    print(f"Using max_iterations = {max_iterations}")

    if verbose > 1:
        # Display the sinogram
        mj.slice_viewer(sinogram, slice_axis=0, title='Original sinogram')

    # Print out model parameters
    ct_model.print_params()

    # Perform Direct reconstruction
    print("\n********** Perform direct reconstruction **************")
    direct_recon = ct_model.direct_recon(sinogram)

    if view_alignment is True:
        # Perform sinogram per-view alignment
        print("\n********** Perform sinogram alignment **************")
        sinogram = mjp.align_sino_views(ct_model, sinogram, direct_recon)

        # Perform direct reconstruction
        print("\n********** Perform direct reconstruction after alignment **************")
        direct_recon = ct_model.direct_recon(sinogram)

    # Weights
    weights = mj.gen_weights(sinogram, weight_type='transmission_root')

    # Perform MBIR reconstruction
    print("\n********** Perform MBIR reconstruction **************")
    mbir_recon, recon_dict = ct_model.recon(sinogram, weights=weights, max_iterations=max_iterations)

    # Save recon to hdf5 FIRST, so the (expensive) result is on disk before anything else runs.
    print("\n*********** save mbir and direct recon in h5 format *************")
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist
    direct_path = os.path.join(output_path, f"zeiss_fdk_recon.h5")
    mj.export_recon_hdf5(direct_path, direct_recon, recon_dict=None)
    mbir_path = os.path.join(output_path, f"zeiss_mbir_recon.h5")
    mj.export_recon_hdf5(mbir_path, mbir_recon, recon_dict=None, remove_flash=True)
    print("Direct recon saved to {}".format(os.path.abspath(direct_path)))
    print("MBIR recon saved to {}".format(os.path.abspath(mbir_path)))

    # Report peak GPU memory for this partition sequence (the point of the experiment).
    # recon() returns a host (numpy) array with the device work already complete, so no
    # block_until_ready is needed (and a numpy array doesn't have that method).
    report_peak_gpu_memory(label=f"(partition_sequence='{partition_sequence_name}')")

    if verbose > 1:
        # Display the results
        mj.slice_viewer(direct_recon, mbir_recon, slice_axis=2, vmin=vmin, vmax=vmax,
                        slice_label=['Direct', 'MBIR'],
                        title='Comparison between Direct and MBIR reconstructions')
