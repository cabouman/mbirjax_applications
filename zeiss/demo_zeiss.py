import os
import sys
import numpy as np
import jax.numpy as jnp
import pprint
import mbirjax as mj
import mbirjax.preprocess as mjp

pp = pprint.PrettyPrinter(indent=4)

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
            'downsample_factor': 2,
            'subsample_view_factor': 2,
            'view_alignment': False,
            'vmin': 0,
            'vmax': 0.4,
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
            'url': '/depot/bouman/data/Zeiss/foam512R1N3000.txrm',
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

    # Load the sinogram and metadata
    print("\n********** Load sinogram and metadata from the data **************")
    sinogram, cone_beam_params, optional_params = mjp.zeiss_cb.compute_sino_and_params(dataset_url, downsample_factor=(downsample_factor, downsample_factor),
                                                                                                 subsample_view_factor=subsample_view_factor)

    # Construct cone beam model
    print("\n********** Construct cone beam model **************")
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    ct_model.set_params(**optional_params)

    # Rerun auto-parameter functions because we changed the assumed detector pitch
    ct_model.auto_set_recon_geometry(sinogram.shape) # Reset default recon shape

    # Sharpness and snr_db
    ct_model.set_params(sharpness=sharpness, snr_db=snr_db, verbose=1)

    if verbose > 1:
        # Display the sinogram
        mj.slice_viewer(sinogram, slice_axis=0, title='Original sinogram')

    # Print out model parameters
    ct_model.print_params()

    # Perform FDK reconstruction
    print("\n********** Perform FDK reconstruction **************")
    direct_recon = ct_model.direct_recon(sinogram)

    if view_alignment is True:
        # Perform sinogram per-view alignment
        print("\n********** Perform sinogram alignment **************")
        sinogram = mjp.align_sino_views(ct_model, sinogram, direct_recon)

        # Perform FDK reconstruction
        print("\n********** Perform FDK reconstruction after alignment **************")
        direct_recon = ct_model.direct_recon(sinogram)

    # Weights
    weights = mj.gen_weights(sinogram, weight_type='transmission_root')

    # Perform MBIR reconstruction
    print("\n********** Perform MBIR reconstruction **************")
    mbir_recon, recon_dict = ct_model.recon(sinogram, weights=weights)

    # Save recon to hdf5
    print("\n*********** save mbir and fdk recon in h5 format *************")
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist
    fdk_path = os.path.join(output_path, f"cone_fdk_recon.h5")
    mj.export_recon_hdf5(fdk_path, direct_recon, recon_dict=None)
    mbir_path = os.path.join(output_path, f"cone_mbir_recon.h5")
    mj.export_recon_hdf5(mbir_path, mbir_recon, recon_dict=None, remove_flash=True)
    print("FDK recon saved to {}".format(os.path.abspath(fdk_path)))
    print("MBIR recon saved to {}".format(os.path.abspath(mbir_path)))

    if verbose > 1:
        # Display the results
        mj.slice_viewer(direct_recon, mbir_recon, slice_axis=2, vmin=vmin, vmax=vmax,
                        slice_label=['FDK', 'MBIR'],
                        title='Comparison between FDK and MBIR reconstructions')
