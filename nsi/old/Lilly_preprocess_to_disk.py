"""Stage 1 of the two-stage NSI workflow: preprocess an NSI scan to a sinogram on disk.

This does ONLY preprocessing -- load + compute the sinogram, crop, clip -- and writes the result
(sinogram + geometry parameters) to an HDF5 file with ``mbirjax.save_preprocessing``.  Reconstruct it
with ``Lilly_recon_from_disk.py``.

Why two stages: (i) preprocessing is expensive and rarely changes, so saving it lets you iterate on
recon parameters without re-preprocessing, and you can inspect/reuse the preprocessed sinogram; (ii)
running recon as a SEPARATE process (a fresh Python invocation) gives the memory-tight recon a clean
GPU allocator with no leftover state from preprocessing's batched GPU work.

Example:
    python Lilly_preprocess_to_disk.py --data_path /path/to/nsi_scan \\
        --output ./output/lilly/lilly_preprocessed.h5
"""
import argparse
import os

import numpy as np
import mbirjax as mj
import mbirjax.preprocess as mjp

if __name__ == "__main__":
    verbose = 1

    parser = argparse.ArgumentParser(description="NSI preprocessing -> sinogram on disk (stage 1 of 2)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the NSI scan directory.")
    parser.add_argument("--output", type=str, default="./output/lilly/lilly_preprocessed.h5",
                        help="Output HDF5 path for the preprocessed sinogram + geometry parameters. "
                             "Point this at a filesystem with room for the full sinogram (large scans "
                             "are tens of GB).")
    parser.add_argument("--downsampling", type=int, default=1,
                        help="Subsampling factor for detector rows and channels.")
    parser.add_argument("--subsample_view_factor", type=int, default=1,
                        help="Subsampling factor for projection views.")
    parser.add_argument("--sino_cropping", type=int, default=1,
                        help="Flag for applying sinogram cropping.")
    args = parser.parse_args()

    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"--data_path does not exist or is not a directory: {args.data_path}")

    downsample_rate = [args.downsampling, args.downsampling]

    if verbose > 0:
        print("\n************** NSI dataset preprocessing **************")
    sino, cone_beam_params, optional_params = mjp.nsi.compute_sino_and_params(
        args.data_path, downsample_factor=downsample_rate, subsample_view_factor=args.subsample_view_factor)

    if bool(args.sino_cropping):
        if verbose > 0:
            print("\n********** Cropping sinogram margins and updating cone-beam geometry parameters **********")
        sino, cone_beam_params, optional_params = mjp.auto_crop_sino_conebeam(sino, cone_beam_params, optional_params)

    # Clip to non-negative ON THE HOST (np.maximum, not jnp.maximum) so the full sinogram is never
    # copied onto a single GPU here -- it is written straight to disk and the recon stage shards it.
    sino = np.maximum(sino, 0.0)

    if verbose > 0:
        print("\n*************** Saving preprocessed sinogram + geometry ***************")
    mjp.save_preprocessing(args.output, sino, cone_beam_params, optional_params)
    if verbose > 0:
        print(f"Preprocessed sinogram shape {tuple(sino.shape)} saved to {os.path.abspath(args.output)}")
        print(f"Reconstruct with:\n  python Lilly_recon_from_disk.py --preprocessed {args.output}")
