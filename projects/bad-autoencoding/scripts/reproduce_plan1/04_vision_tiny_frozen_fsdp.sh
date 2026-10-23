#!/usr/bin/env bash
# Full PLAN_1 vision tiny frozen run using GPU-only FSDP.
#
# The defaults target two 16 GB GPUs:
#   batch 1 x grad-accum 24 x 2 ranks = effective batch 48
#
# All model/gradient/optimizer/activation memory stays on the GPUs; this
# launcher never enables CPU offload. If the probe does not fit, use the
# single-GPU DGX run or validation-only instead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

NPROC="${NPROC:-2}"
FSDP_BATCH_SIZE="${FSDP_BATCH_SIZE:-1}"
FSDP_GRAD_ACCUM="${FSDP_GRAD_ACCUM:-24}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"
OUTPUT_DIR="${VISION_FSDP_OUTPUT_DIR:-$OUTPUT_ROOT/vision_tiny_frozen_fsdp}"

EFFECTIVE=$((NPROC * FSDP_BATCH_SIZE * FSDP_GRAD_ACCUM))
if [[ "$EFFECTIVE" -ne 48 ]]; then
    echo "ERROR: effective batch is $EFFECTIVE, expected 48."
    echo "Set NPROC=$NPROC, FSDP_BATCH_SIZE=$FSDP_BATCH_SIZE, FSDP_GRAD_ACCUM=$FSDP_GRAD_ACCUM."
    exit 1
fi

if [[ "$OPTIMIZER" == "adafactor" ]]; then
    echo "WARNING: OPTIMIZER=adafactor is a memory fallback, NOT the paper's AdamW recipe."
    echo "         Results must be reported as non-faithful to the paper."
fi

if [[ ! -f "$TRAIN_DATA/dataset_info.json" || ! -f "$VAL_DATA/dataset_info.json" ]]; then
    echo "ERROR: Arrow datasets not found."
    echo "Expected: $TRAIN_DATA and $VAL_DATA"
    exit 1
fi
if [[ -e "$OUTPUT_DIR" && "${FORCE:-0}" != "1" ]]; then
    echo "ERROR: output directory already exists: $OUTPUT_DIR"
    echo "Set FORCE=1 to delete it, or VISION_FSDP_OUTPUT_DIR=<new path> to keep it."
    exit 1
fi
if [[ "${FORCE:-0}" == "1" ]]; then
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

echo "==> FSDP vision tiny frozen run"
echo "    output:    $OUTPUT_DIR"
echo "    ranks:     $NPROC"
echo "    batch:     $FSDP_BATCH_SIZE x $FSDP_GRAD_ACCUM x $NPROC = $EFFECTIVE"
echo "    optimizer: $OPTIMIZER"
echo "    cpu offload: disabled"

torchrun --nproc_per_node="$NPROC" train_fsdp.py \
    --regime vision \
    --vision_mode tiny \
    --vision_prompt free_ocr \
    --no-train-encoder \
    --objective lm \
    --data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$FSDP_BATCH_SIZE" \
    --gradient_accumulation_steps "$FSDP_GRAD_ACCUM" \
    --target_effective_batch 48 \
    --optimizer "$OPTIMIZER" \
    --use_decoder_checkpointing \
    --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --eval_steps "${EVAL_STEPS:-500}" \
    --save_steps 0 \
    --log_steps 10 \
    --num_qualitative_samples 0 \
    --save_val_per_sample \
    --num_workers "${NUM_WORKERS:-4}" \
    --prefetch_factor "${PREFETCH_FACTOR:-2}" \
    --initial_validation \
    --seed "${SEED:-42}"
