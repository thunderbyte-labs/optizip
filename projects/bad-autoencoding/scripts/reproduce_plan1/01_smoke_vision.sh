#!/usr/bin/env bash
# Short vision smoke test: validates model loading, image pipeline, eager
# attention fallback and the 78-token context (73 vision + 4 prompt + BOS).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

OUTPUT_DIR="$OUTPUT_ROOT/smoke_vision"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "==> Vision smoke test"
uv run python train.py \
    --regime vision \
    --vision_mode tiny \
    --vision_prompt free_ocr \
    --no-train-encoder \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
    --objective lm \
    --data_path "$TRAIN_DATA" \
    --val_data_path "$SMOKE_VAL" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples 64 \
    --batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_epochs 1 \
    --eval_steps 4 \
    --save_steps 0 \
    --log_steps 1 \
    --num_qualitative_samples 0 \
    --num_workers 0 \
    --no_checkpoints \
    --seed "$SEED"

echo
echo "==> Validating logged context length"
grep -q "Effective context tokens (per-sample): 78" "$OUTPUT_DIR/train.log" || {
    echo "ERROR: expected 78 context tokens (73 vision + 4 prompt + BOS)."
    grep "Effective context tokens" "$OUTPUT_DIR/train.log" || true
    exit 1
}
grep -q "Target tokens per sample: 1000" "$OUTPUT_DIR/train.log" || {
    echo "ERROR: expected 1000 target tokens per sample."
    exit 1
}

echo "OK: vision smoke test passed (78 context tokens, 1000 target tokens)."
