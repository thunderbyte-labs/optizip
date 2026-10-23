#!/usr/bin/env bash
# One-step GPU-only FSDP memory probe.
#
# This is the go/no-go gate before a multi-day run. It performs one
# forward/backward/optimizer step, writes probe_result.json, and exits.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

NPROC="${NPROC:-2}"
FSDP_BATCH_SIZE="${FSDP_BATCH_SIZE:-1}"
FSDP_GRAD_ACCUM="${FSDP_GRAD_ACCUM:-24}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"
PROBE_SAMPLES="${PROBE_SAMPLES:-8}"
OUTPUT_DIR="${PROBE_OUTPUT_DIR:-$OUTPUT_ROOT/fsdp_probe}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    read -r -a PY <<< "$PYTHON_BIN"
else
    PY=(uv run python)
fi

EFFECTIVE=$((NPROC * FSDP_BATCH_SIZE * FSDP_GRAD_ACCUM))
if [[ "$EFFECTIVE" -ne 48 ]]; then
    echo "ERROR: effective batch is $EFFECTIVE, expected 48."
    exit 1
fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "==> FSDP probe: $NPROC ranks, batch $FSDP_BATCH_SIZE x accum $FSDP_GRAD_ACCUM"
echo "    optimizer: $OPTIMIZER, samples: $PROBE_SAMPLES, cpu offload: disabled"

"$TORCHRUN_BIN" --nproc_per_node="$NPROC" train_fsdp.py \
    --probe \
    --regime vision \
    --vision_mode tiny \
    --vision_prompt free_ocr \
    --no-train-encoder \
    --objective lm \
    --data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples "$PROBE_SAMPLES" \
    --batch_size "$FSDP_BATCH_SIZE" \
    --gradient_accumulation_steps "$FSDP_GRAD_ACCUM" \
    --target_effective_batch 48 \
    --optimizer "$OPTIMIZER" \
    --use_decoder_checkpointing \
    --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
    --num_epochs 1 \
    --eval_steps 0 \
    --save_steps 0 \
    --num_qualitative_samples 0 \
    --num_workers 0 \
    --no_checkpoints \
    --probe_memory_limit_gb "${PROBE_MEMORY_LIMIT_GB:-0}"

echo
echo "Probe result: $OUTPUT_DIR/probe_result.json"
"${PY[@]}" scripts/reproduce_plan1/gpu_doctor.py feasibility \
    --probe-json "$OUTPUT_DIR/probe_result.json"
