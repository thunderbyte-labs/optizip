#!/usr/bin/env bash
# Validate a checkpoint on a new machine/GPU without training.
#
# Generic usage:
#   REGIME=vision CHECKPOINT=/path/best_checkpoint.pt \
#   VAL_DATA=/path/val_arrow OUTPUT_DIR=outputs/validation_vision \
#   bash scripts/reproduce_plan1/06_validate_checkpoint.sh
#
#   REGIME=conv1d_residual COMPRESSION_TARGET=63 CHECKPOINT=... \
#   VAL_DATA=... bash scripts/reproduce_plan1/06_validate_checkpoint.sh
#
# Writes val_per_sample.jsonl for estimate_paper_metric.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

REGIME="${REGIME:-vision}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the checkpoint to validate}"
OUTPUT_DIR="${VALIDATION_OUTPUT_DIR:-$OUTPUT_ROOT/validation_${REGIME}}"
BATCH_SIZE="${VALIDATION_BATCH_SIZE:-1}"
NUM_WORKERS="${VALIDATION_NUM_WORKERS:-2}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    exit 1
fi
if [[ ! -f "$VAL_DATA/dataset_info.json" ]]; then
    echo "ERROR: validation Arrow dataset not found: $VAL_DATA"
    exit 1
fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

COMMON_ARGS=(
    --objective lm
    --val_data_path "$VAL_DATA"
    --validation_only
    --init_from_checkpoint "$CHECKPOINT"
    --allow_objective_switch
    --attn_implementation "${ATTN_IMPLEMENTATION:-eager}"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --num_qualitative_samples 0
    --save_val_per_sample
    --output_dir "$OUTPUT_DIR"
)

if [[ "$REGIME" == "vision" ]]; then
    COMMON_ARGS+=(
        --regime vision
        --vision_mode "${VISION_MODE:-tiny}"
        --vision_prompt free_ocr
        --no-train-encoder
    )
elif [[ "$REGIME" == "conv1d_residual" ]]; then
    COMMON_ARGS+=(
        --regime conv1d_residual
        --compression_target "${COMPRESSION_TARGET:-63}"
        --conv_kernel "${CONV_KERNEL:-5}"
        --no-train-encoder
    )
else
    COMMON_ARGS+=(--regime "$REGIME")
fi

echo "==> Validation-only: $REGIME"
echo "    checkpoint: $CHECKPOINT"
echo "    val data:   $VAL_DATA"
echo "    output:     $OUTPUT_DIR"

uv run python train.py "${COMMON_ARGS[@]}"
