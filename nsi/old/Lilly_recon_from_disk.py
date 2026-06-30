"""Stage 2 of the two-stage NSI workflow: reconstruct from a preprocessed sinogram on disk.

Loads the sinogram + geometry parameters written by ``Lilly_preprocess_to_disk.py``
(``mbirjax.save_preprocessing``), rebuilds the cone-beam model, and runs the reconstruction.  Weights
are recomputed here from the sinogram (cheap, one pass) rather than saved.

Run this as its own process so the memory-tight recon starts with a clean GPU allocator (no leftover
state from preprocessing's batched GPU work).

Example:
    python Lilly_recon_from_disk.py --preprocessed ./output/lilly/lilly_preprocessed.h5 --num_metal 0
"""
import argparse
import os

import mbirjax as mj
import mbirjax.preprocess as mjp

if __name__ == "__main__":
    sharpness = 1.0
    verbose = 1

    parser = argparse.ArgumentParser(description="Reconstruct from a preprocessed sinogram (stage 2 of 2)")
    parser.add_argument("--preprocessed", type=str, required=True,
                        help="Path to the HDF5 file written by Lilly_preprocess_to_disk.py.")
    parser.add_argument("--output_path", type=str, default="./output/lilly/",
                        help="Directory for the output recon. Point at a filesystem with room for the "
                             "full reconstruction (large recons are tens of GB).")
    parser.add_argument("--num_metal", type=int, default=2,
                        help="Number of metal types for segmentation and MAR (0 = standard MBIR recon).")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    if verbose > 0:
        print("\n*************** Loading preprocessed sinogram + geometry ***************")
    sino, cone_beam_params, optional_params, weights_trans = mjp.load_preprocessing(args.preprocessed)

    if verbose > 0:
        print("\n***************** Set up MBIRJAX model ****************")
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    ct_model.set_params(**optional_params)
    ct_model.set_params(sharpness=sharpness, verbose=verbose, positivity_flag=True)
    # Use the custom weights if they were saved; otherwise regenerate the standard ones (cheap).
    if weights_trans is None:
        weights_trans = mj.gen_weights(sino, weight_type='transmission_root')
        if verbose > 0:
            print("No saved weights; regenerated 'transmission_root' weights.")
    elif verbose > 0:
        print("Using custom weights loaded from the preprocessed file.")
    if verbose > 0:
        ct_model.print_params()

    if verbose > 0:
        print("\n*************** Compute reconstruction ***************")
    recon = mjp.recon_plastic_metal(ct_model, sino, weights_trans, num_metal=args.num_metal, verbose=verbose)

    # Voxel pitch for the output filename
    delta_voxel_um = ct_model.get_params('delta_voxel') * ct_model.get_params('alu_value') * 1000

    dataset_tag = os.path.splitext(os.path.basename(args.preprocessed))[0]
    if verbose > 0:
        print("\n*********** Save recon in h5 format *************")
    recon_path = os.path.join(
        args.output_path,
        f"recon_{dataset_tag}_nummetal_{args.num_metal}_voxel_pitch_{delta_voxel_um:.2f}um_mar.h5")
    mj.export_recon_hdf5(recon_path, recon, recon_dict=None, remove_flash=True)
    if verbose > 0:
        print("Reconstruction saved to {}".format(os.path.abspath(recon_path)))
