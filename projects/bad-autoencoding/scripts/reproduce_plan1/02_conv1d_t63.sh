#!/usr/bin/env bash
# Full PLAN_1 run 1/2: Hierarchical (conv1d_residual) t=63, LM objective.
#
# Paper row (Table 2): Hierarchical t=63, 65 tokens, 15.4x, PPL 5.14.
# Encoder and decoder are initialized from the released reconstruction
# checkpoint, then the frozen encoder is kept and only the decoder is trained
# (same recipe as the authors' released t=250 LM run, which reached 4.96).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

OUTPUT_DIR="${CONV_OUTPUT_DIR:-$OUTPUT_ROOT/conv1d_t63_lm_recon-init}"

if [[ ! -f "$RECON_CKPT" ]]; then
    echo "ERROR: reconstruction checkpoint not found: $RECON_CKPT"
    echo "Run scripts/reproduce_plan1/00_setup_gb10.sh first."
    exit 1
fi
if [[ ! -f "$TRAIN_DATA/dataset_info.json" || ! -f "$VAL_DATA/dataset_info.json" ]]; then
    echo "ERROR: Arrow datasets not found."
    echo "Expected: $TRAIN_DATA and $VAL_DATA"
    echo "Run scripts/reproduce_plan1/00_setup_gb10.sh first."
    exit 1
fi
if [[ -e "$OUTPUT_DIR" && "${FORCE:-0}" != "1" ]]; then
    echo "ERROR: output directory already exists: $OUTPUT_DIR"
    echo "Set FORCE=1 to delete it, or CONV_OUTPUT_DIR=<new path> to keep it."
    exit 1
fi
if [[ "${FORCE:-0}" == "1" ]]; then
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

echo "==> Full run: conv1d_residual t=63 (LM, recon init)"
echo "    output: $OUTPUT_DIR"
echo "    attn:   $ATTN_IMPLEMENTATION"
echo "    batch:  $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"

uv run python train.py \
    --regime conv1d_residual \
    --compression_target 63 \
    --conv_kernel 5 \
    --objective lm \
    --data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --init_from_checkpoint "$RECON_CKPT" \
    --allow_objective_switch \
    --no-train-encoder \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
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
