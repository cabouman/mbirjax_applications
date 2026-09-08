#!/bin/bash
#
# Demo run script for 4D MACE CT reconstruction.
#
# Instructions:
#   1. Set DATA_PATH to the extracted NSI dataset directory.
#   2. Optionally adjust the flags below.
#   3. Run:  bash test_script_4d.sh

DATA_PATH=/depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024/

mkdir -p ~/mbirjax_notes/


PYTHONUNBUFFERED=1 python Lilly_recon_4d.py \
  --data_path           "$DATA_PATH" \
  --downsampling        1 \
  --max_mace_iterations 10 \
  2>&1 | tee ~/mbirjax_notes/Lilly_4d_ds1_run.log

# The views are not subsampled for 4D data, in order to preserve recon quality.

# For a quick test, reconstruct only the first N time frames by adding:
#   --num_frames 25 \

# Advanced parameters:
#   --output_path ./output/lilly # Directory for the recon, GIF and init cache.
#   --frames_per_rotation 6      # Time frames per 360 degrees.  Must match the gating geometry.
#   --frame_overlap_factor 2.0   # Number of frames that share any given view.  At 1.0 the
#                                # frames do not overlap.  At the default 2.0 every view is
#                                # shared by two frames, so consecutive frames overlap by 50%.
#                                # Larger values give each frame more views at the cost of
#                                # more motion blur.
