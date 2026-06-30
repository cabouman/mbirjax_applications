#!/usr/bin/env bash
# Two-stage NSI reconstruction: preprocess to disk, then reconstruct from disk in a SEPARATE process.
#
# Running stage 2 as its own `python` invocation is the point of this script: the memory-tight recon
# starts with a clean GPU allocator, with no leftover state from preprocessing's batched GPU work.
# (An in-process orchestrator that imported and called both stages would lose that benefit.)
#
# Usage:
#   ./Lilly_two_stage.sh <data_path> [output_dir] [num_metal]
# Example:
#   ./Lilly_two_stage.sh /path/to/nsi_scan ./output/lilly 0
set -euo pipefail

DATA_PATH="${1:?Usage: $0 <data_path> [output_dir] [num_metal]}"
OUTPUT_DIR="${2:-./output/lilly}"
NUM_METAL="${3:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPROCESSED="${OUTPUT_DIR}/lilly_preprocessed.h5"

mkdir -p "${OUTPUT_DIR}"

echo "=== Stage 1/2: preprocess -> ${PREPROCESSED} ==="
python "${SCRIPT_DIR}/Lilly_preprocess_to_disk.py" \
    --data_path "${DATA_PATH}" \
    --output "${PREPROCESSED}"

echo "=== Stage 2/2: reconstruct from ${PREPROCESSED} (fresh process) ==="
python "${SCRIPT_DIR}/Lilly_recon_from_disk.py" \
    --preprocessed "${PREPROCESSED}" \
    --output_path "${OUTPUT_DIR}" \
    --num_metal "${NUM_METAL}"

echo "=== Done.  Recon written under ${OUTPUT_DIR} ==="
