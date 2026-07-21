#!/bin/bash

# Test script for large recons: MBIR with no metal
# Uses the mbirjax default partition sequence.

DATA_PATH=/depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal/

mkdir -p ~/mbirjax_notes/
python Lilly_recon.py \
  --data_path "$DATA_PATH" \
  --downsampling 1 \
  --subsample_view_factor 2 \
  --num_metal 0 \
  --sino_cropping 1 \
  --max_iterations 15 \
  2>&1 | tee ~/mbirjax_notes/Lilly_no_metal_ds1_run.log
