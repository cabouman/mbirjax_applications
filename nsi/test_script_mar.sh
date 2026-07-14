#!/bin/bash

# Test script for large recons: MAR with num_metal=2

DATA_PATH=/depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal/
PSEQ=skip_0

echo "==================== partition_sequence: $PSEQ ===================="
mkdir -p ~/mbirjax_notes/
python Lilly_recon.py \
  --data_path "$DATA_PATH" \
  --downsampling 1 \
  --subsample_view_factor 2 \
  --num_metal 2 \
  --sino_cropping 1 \
  --partition_sequence $PSEQ \
  --max_iterations 15 \
  2>&1 | tee ~/mbirjax_notes/Lilly_mar_ds1_run.log
