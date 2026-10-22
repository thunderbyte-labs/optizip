#!/usr/bin/env bash
# Prepare a GB10 (aarch64 + CUDA 13) machine for the PLAN_1 verification runs.
#
# This script is idempotent: existing weights, checkpoints and Arrow datasets
# are reused. Run it from anywhere; paths are resolved from the repository root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

echo "==> Repository: $REPO_ROOT"
echo "==> Data dir:   $DATA_DIR"
echo "==> Output root: $OUTPUT_ROOT"

echo
echo "==> Checking PyTorch / CUDA"
uv run python - <<'PY'
import torch

print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available. Run `uv sync` first (torch must be the CUDA build)."
    )
print(f"device={torch.cuda.get_device_name(0)}")
PY

echo
echo "==> Downloading DeepSeek-OCR weights (~6.7 GB, skipped if cached)"
uv run python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download("deepseek-ai/DeepSeek-OCR")
print(f"weights: {path}")
PY

echo
echo "==> Downloading conv1d t=63 reconstruction checkpoint (~18.7 GB)"
if [[ -f "$RECON_CKPT" ]]; then
    echo "already present: $RECON_CKPT"
else
    mkdir -p "$CKPT_DIR"
    uv run hf download ivnle/bad-autoencoding \
        conv1d_t63_h0_recon/best_checkpoint.pt \
        --local-dir "$CKPT_DIR"
fi

echo
echo "==> Converting JSONL splits to Arrow (memory-mapped)"
for split in train val; do
    arrow_dir="$DATA_DIR/${split}_arrow"
    if [[ -f "$arrow_dir/dataset_info.json" ]]; then
        echo "already present: $arrow_dir"
    else
        uv run python data/scripts/convert_to_arrow.py \
            --input "$DATA_DIR/${split}.jsonl" \
            --output "$arrow_dir"
    fi
done

echo
echo "==> Creating smoke validation split (32 samples)"
uv run python - <<'PY'
import os
from pathlib import Path

from datasets import load_from_disk

src = Path(os.environ["VAL_DATA"])
dst = Path(os.environ["SMOKE_VAL"])
if (dst / "dataset_info.json").exists():
    print(f"already present: {dst}")
else:
    load_from_disk(str(src)).select(range(32)).save_to_disk(str(dst))
    print(f"created: {dst}")
PY

echo
echo "==> Setup complete."
echo "Next: bash scripts/reproduce_plan1/01_smoke_vision.sh"
