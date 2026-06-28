#!/bin/bash

python Lilly_recon.py \
  --data_path /depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal/ \
  --downsampling 1 \
  --subsample_view_factor 2 \
  --num_metal 0 \
  --sino_cropping 1
  2>&1 | tee ~/mbirjax_notes/Lilly_no_metal_ds1_run.log

