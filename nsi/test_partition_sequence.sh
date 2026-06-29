#!/bin/bash

# Test a predefined partition sequence for large recons.
# Available --partition_sequence values: default, coarse_4_128, slow_start, slow_dip
python Lilly_recon_partition_sequence.py \
  --data_path /depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal/ \
  --downsampling 1 \
  --subsample_view_factor 2 \
  --num_metal 1 \
  --sino_cropping 1 \
  --partition_sequence coarse_4_128 \
  2>&1 | tee ~/mbirjax_notes/Lilly_pseq_coarse_4_128_run.log
