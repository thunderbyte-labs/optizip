#!/usr/bin/env bash
# Run both PLAN_1 full configurations sequentially and record their status.
#
# Usage:
#   bash scripts/reproduce_plan1/run_all.sh
#
# A failure in one run does not stop the other; the wrapper exits non-zero if
# either run failed. Logs are written by train.py inside each output directory;
# redirect this script's stdout to a file to keep a wrapper log.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

mkdir -p "$OUTPUT_ROOT"

status=0
echo "=== PLAN_1 run-all started $(date -Is) ==="
echo "DATA_DIR=$DATA_DIR"
echo "CKPT_DIR=$CKPT_DIR"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION BATCH_SIZE=$BATCH_SIZE GRAD_ACCUM=$GRAD_ACCUM"

for script in 02_conv1d_t63.sh 03_vision_tiny_frozen.sh; do
    echo
    echo "=== START $script $(date -Is) ==="
    if bash "$SCRIPT_DIR/$script"; then
        echo "=== OK $script $(date -Is) ==="
    else
        echo "=== FAIL $script $(date -Is) ==="
        status=1
    fi
done

echo
echo "=== PLAN_1 run-all finished $(date -Is) (status=$status) ==="
exit "$status"
