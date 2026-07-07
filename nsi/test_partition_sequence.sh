#!/bin/bash

# Test predefined partition sequences for large recons.
# Available --partition_sequence values: default, coarse_4_128, slow_start, slow_dip
# mbirjax writes its detailed recon log to ~/mbirjax_notes/recon_<dataset>_pseq_<name>.log;
# the tee below additionally captures console output + any Python tracebacks per run.

DATA_PATH=/depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal/
LOG_DIR=~/mbirjax_notes
mkdir -p "$LOG_DIR"

for PSEQ in default skip_0; do
  echo "==================== partition_sequence: $PSEQ ===================="
  python Lilly_recon_partition_sequence.py \
    --data_path "$DATA_PATH" \
    --downsampling 1 \
    --subsample_view_factor 2 \
    --num_metal 0 \
    --sino_cropping 1 \
    --partition_sequence "$PSEQ" \
    2>&1 | tee "$LOG_DIR/console_pseq_${PSEQ}.log"
done
