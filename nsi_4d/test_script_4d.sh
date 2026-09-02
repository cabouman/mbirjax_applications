#!/bin/bash
#
# 4D MACE CT Reconstruction — demo run script.
#
# Instructions:
#   1. Set DATA_PATH to the extracted NSI dataset directory.
#   2. Optionally adjust the flags below.
#   3. Run:  bash test_script_4d.sh
#

cd "$(dirname "${BASH_SOURCE[0]}")"

DATA_PATH=/depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024/
OUTPUT_PATH=./output

mkdir -p "$OUTPUT_PATH"
mkdir -p ~/4dct_logs/

# PYTHONUNBUFFERED: this pipes into `tee`, and Python block-buffers stdout when it is a pipe
# rather than a terminal, so without it nothing appears in the log until ~8 KB has accumulated.
# Preprocessing and the MBIR init print little, so the log stays empty for the first half hour
# of a multi-hour run and looks hung.  (stderr is never block-buffered, which is why warnings
# still show up.)  `python -u` does the same thing but is lost across an exec.
PYTHONUNBUFFERED=1 python Lilly_recon.py \
  --data_path           "$DATA_PATH" \
  --output_path         "$OUTPUT_PATH" \
  --downsampling        1 \
  --max_mace_iterations 10 \
  2>&1 | tee ~/4dct_logs/recon_4d_run.log

# Quick test - reconstruct only the first N time frames, add:
#   --num_frames 25 \
#
# Advanced (leave at defaults unless you know why):
#   --frames_per_rotation 6      # time frames per 360 deg; must match the gating geometry
#   --frame_overlap_factor 2.0   # frames sharing any given view (MACE tuning)