#!/usr/bin/env bash
# Full PLAN_1 run 2/2: Vision tiny with a frozen encoder, LM objective.
#
# Paper row (Table 2): Vision tiny frozen, 78 tokens (73 vision + 4-token
# "Free OCR." prompt + BOS), 12.8x, PPL 5.35.
# The decoder is the pretrained DeepSeek-OCR decoder; only it is trained.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

OUTPUT_DIR="${VISION_OUTPUT_DIR:-$OUTPUT_ROOT/vision_tiny_frozen_lm}"

if [[ ! -f "$TRAIN_DATA/dataset_info.json" || ! -f "$VAL_DATA/dataset_info.json" ]]; then
    echo "ERROR: Arrow datasets not found."
    echo "Expected: $TRAIN_DATA and $VAL_DATA"
    echo "Run scripts/reproduce_plan1/00_setup_gb10.sh first."
    exit 1
fi
if [[ -e "$OUTPUT_DIR" && "${FORCE:-0}" != "1" ]]; then
    echo "ERROR: output directory already exists: $OUTPUT_DIR"
    echo "Set FORCE=1 to delete it, or VISION_OUTPUT_DIR=<new path> to keep it."
    exit 1
fi
if [[ "${FORCE:-0}" == "1" ]]; then
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

echo "==> Full run: vision tiny frozen (LM)"
echo "    output: $OUTPUT_DIR"
echo "    attn:   $ATTN_IMPLEMENTATION"
echo "    batch:  $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"

uv run python train.py \
    --regime vision \
    --vision_mode tiny \
    --vision_prompt free_ocr \
    --no-train-encoder \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
    --objective lm \
    --data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps 0 \
    --log_steps 10 \
    --num_qualitative_samples 0 \
    --save_val_per_sample \
    --num_workers "$NUM_WORKERS" \
    --prefetch_factor "$PREFETCH_FACTOR" \
    --initial_validation \
    --seed "$SEED"
