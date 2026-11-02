#!/usr/bin/env bash
# PLAN_2: Vision tiny frozen with DeepSeek-OCR-2 (DeepEncoder V2).
#
# Reproduces the PLAN_1 "vision tiny frozen" recipe but swaps the OCR-1 vision
# encoder + decoder for DeepSeek-OCR-2. DeepEncoder V2 has no native 512px /
# 78-token mode, so we add an 8x8 causal-flow query embedding (query_512)
# initialised by bicubic interpolation of the pretrained query_1024 grid.
# This yields 64 visual queries + 1 view separator = 65 image tokens, run on the
# existing 512x512 "tiny" images.
#
# Reference: OCR-1 vision tiny frozen (PPL 5.35 paper / 5.45 reproduced).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

OUTPUT_DIR="${OCR2_VISION_OUTPUT_DIR:-$OUTPUT_ROOT/vision_tiny_frozen_ocr2_lm}"

if [[ ! -f "$TRAIN_DATA/dataset_info.json" || ! -f "$VAL_DATA/dataset_info.json" ]]; then
    echo "ERROR: Arrow datasets not found."
    echo "Expected: $TRAIN_DATA and $VAL_DATA"
    echo "Set DATA_DIR to the directory containing train_arrow / val_arrow / images."
    exit 1
fi
if [[ -e "$OUTPUT_DIR" && "${FORCE:-0}" != "1" ]]; then
    echo "ERROR: output directory already exists: $OUTPUT_DIR"
    echo "Set FORCE=1 to delete it, or OCR2_VISION_OUTPUT_DIR=<new path> to keep it."
    exit 1
fi
if [[ "${FORCE:-0}" == "1" ]]; then
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

echo "==> Full run: vision tiny frozen with DeepSeek-OCR-2 (LM)"
echo "    output:  $OUTPUT_DIR"
echo "    data:    $TRAIN_DATA"
echo "    attn:    $ATTN_IMPLEMENTATION"
echo "    batch:   $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo "    encoder: frozen (query_512 interpolated from query_1024)"

uv run python train.py \
    --regime vision \
    --vision_mode tiny \
    --ocr_version ocr2 \
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
