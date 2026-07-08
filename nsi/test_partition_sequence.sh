#!/bin/bash

# Test script for large recons

DATA_PATH=/depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal/

for PSEQ in default skip_0; do
  echo "==================== partition_sequence: $PSEQ ===================="
  python Lilly_recon_partition_sequence.py \
    --data_path "$DATA_PATH" \
    --downsampling 1 \
    --subsample_view_factor 2 \
    --num_metal 0 \
    --sino_cropping 1 \
    --partition_sequence "$PSEQ" \
    --max_iterations 15
done
