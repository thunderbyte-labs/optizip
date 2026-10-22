#!/usr/bin/env bash
# Shared configuration for the PLAN_1 reproduction scripts.
# Source this file from the other scripts; do not execute it directly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Data
DATA_DIR="${DATA_DIR:-./data/training/full}"
TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/train_arrow}"
VAL_DATA="${VAL_DATA:-$DATA_DIR/val_arrow}"
SMOKE_VAL="${SMOKE_VAL:-$DATA_DIR/val_arrow_smoke}"

# Checkpoints
CKPT_DIR="${CKPT_DIR:-./checkpoints/bad-autoencoding}"
RECON_CKPT="${RECON_CKPT:-$CKPT_DIR/conv1d_t63_h0_recon/best_checkpoint.pt}"

# Training
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-6}"
EVAL_STEPS="${EVAL_STEPS:-500}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
SEED="${SEED:-42}"

# Attention backend. The paper used FlashAttention-2, but there is no
# aarch64 + CUDA 13 wheel for flash-attn. Eager is the validated fallback.
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"

export DATA_DIR TRAIN_DATA VAL_DATA SMOKE_VAL CKPT_DIR RECON_CKPT
export OUTPUT_ROOT BATCH_SIZE GRAD_ACCUM EVAL_STEPS NUM_WORKERS PREFETCH_FACTOR SEED
export ATTN_IMPLEMENTATION
