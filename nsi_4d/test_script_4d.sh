#!/bin/bash
#
# 4D MACE CT Reconstruction — demo run script.
#
# Instructions:
#   1. Set DATA_PATH to the extracted NSI dataset directory.
#   2. Optionally adjust the flags below.
#   3. Run:  bash test_script_4d.sh
#

DATA_PATH=/depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024/

mkdir -p ~/mbirjax_notes/


PYTHONUNBUFFERED=1 python Lilly_recon_4d.py \
  --data_path           "$DATA_PATH" \
  --downsampling        1 \
  --max_mace_iterations 10 \
  2>&1 | tee ~/mbirjax_notes/Lilly_4d_ds1_run.log

# Quick test - reconstruct only the first N time frames, add:
#   --num_frames 25 \
#   No view subsampling here for 4D data to ensure better recon quality
# Advanced (leave at defaults unless you know why):
#   --output_path ./output/lilly # where the recon, GIF and init cache are written
#   --gif_slice_axis 1           # write just one GIF plane: 0=time, 1=x, 2=y, 3=z
#                                # (default writes all three spatial planes)
#   --gif_slice_index 130        # index along that axis; default is the middle
#   --frames_per_rotation 6      # time frames per 360 deg; must match the gating geometry
#   --frame_overlap_factor 2.0   # frames sharing any given view (MACE tuning)