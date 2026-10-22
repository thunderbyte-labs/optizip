#!/usr/bin/env bash
# Short conv1d smoke test: validates the reconstruction checkpoint loading,
# eager attention fallback and the 65-token context (63 + separator + BOS).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

if [[ ! -f "$RECON_CKPT" ]]; then
    echo "ERROR: reconstruction checkpoint not found: $RECON_CKPT"
    echo "Run scripts/reproduce_plan1/00_setup_gb10.sh first."
    exit 1
fi

OUTPUT_DIR="$OUTPUT_ROOT/smoke_conv1d"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "==> Conv1d t=63 smoke test"
uv run python train.py \
    --regime conv1d_residual \
    --compression_target 63 \
    --conv_kernel 5 \
    --objective lm \
    --data_path "$TRAIN_DATA" \
    --val_data_path "$SMOKE_VAL" \
    --output_dir "$OUTPUT_DIR" \
    --init_from_checkpoint "$RECON_CKPT" \
    --allow_objective_switch \
    --no-train-encoder \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
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
grep -q "Effective context tokens (per-sample): 65" "$OUTPUT_DIR/train.log" || {
    echo "ERROR: expected 65 context tokens (63 + separator + BOS)."
    grep "Effective context tokens" "$OUTPUT_DIR/train.log" || true
    exit 1
}
grep -q "Target tokens per sample: 1000" "$OUTPUT_DIR/train.log" || {
    echo "ERROR: expected 1000 target tokens per sample."
    exit 1
}

echo "OK: conv1d smoke test passed (65 context tokens, 1000 target tokens)."
