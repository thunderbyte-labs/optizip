#!/usr/bin/env python3
"""Extract PLAN_1 verification results from training logs.

Example:
    uv run python scripts/reproduce_plan1/extract_results.py
    uv run python scripts/reproduce_plan1/extract_results.py \
        --conv-run outputs/conv1d_t63_lm_recon-init \
        --vision-run outputs/vision_tiny_frozen_lm
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

DEFAULT_RUNS = {
    "conv1d_residual": {
        "dirname": "conv1d_t63_lm_recon-init",
        "paper_config": "hierarchical t=63",
        "paper_ppl": 5.14,
    },
    "vision": {
        "dirname": "vision_tiny_frozen_lm",
        "paper_config": "vision tiny frozen",
        "paper_ppl": 5.35,
    },
}

RE_LOADED = re.compile(r"Loaded ([\d,]+) samples from")
RE_CONTEXT = re.compile(r"Effective context tokens \(per-sample\): (\d+)")
RE_TARGET = re.compile(r"Target tokens per sample: (\d+)")
RE_VAL = re.compile(r"Validation loss: ([\d.]+), perplexity: ([\d.]+)")


def parse_args_json(run_dir: Path) -> dict:
    path = run_dir / "args.json"
    if not path.exists():
        raise FileNotFoundError(f"args.json not found in {run_dir}")
    with open(path) as f:
        return json.load(f)


def parse_log(run_dir: Path) -> dict:
    path = run_dir / "train.log"
    if not path.exists():
        raise FileNotFoundError(f"train.log not found in {run_dir}")
    text = path.read_text(errors="replace")

    loaded = RE_LOADED.search(text)
    context = RE_CONTEXT.search(text)
    target = RE_TARGET.search(text)
    val_matches = RE_VAL.findall(text)

    if not loaded:
        raise ValueError(f"could not find the training sample count in {path}")
    if not context:
        raise ValueError(f"could not find the effective context length in {path}")
    if not target:
        raise ValueError(f"could not find the target token count in {path}")
    if not val_matches:
        raise ValueError(f"could not find a validation loss in {path}")

    final_loss, final_ppl = (float(x) for x in val_matches[-1])
    return {
        "num_samples": int(loaded.group(1).replace(",", "")),
        "context_tokens": int(context.group(1)),
        "target_tokens": int(target.group(1)),
        "final_val_loss": final_loss,
        "final_val_perplexity": final_ppl,
    }


def checkpoint_path(run_dir: Path) -> Path:
    best = run_dir / "best_checkpoint.pt"
    final = run_dir / "final_checkpoint.pt"
    if best.exists():
        return best
    if final.exists():
        return final
    raise FileNotFoundError(f"no checkpoint found in {run_dir}")


def infer_init(args: dict) -> str:
    init = args.get("init_from_checkpoint")
    if not init:
        return "DS"
    return "recon" if "recon" in str(init).lower() else "checkpoint"


def format_run(run_dir: Path, spec: dict) -> str:
    args = parse_args_json(run_dir)
    metrics = parse_log(run_dir)

    effective_batch = int(args["batch_size"]) * int(args["gradient_accumulation_steps"])
    tokens_seen = metrics["num_samples"] * (
        metrics["context_tokens"] + metrics["target_tokens"]
    )
    compression_ratio = 1000.0 / metrics["context_tokens"]

    if args["regime"] == "conv1d_residual":
        config = f"compression_target={args['compression_target']}"
    elif args["regime"] == "vision":
        config = f"vision_mode={args['vision_mode']}"
    else:
        config = args["regime"]

    precise_ppl = math.exp(metrics["final_val_loss"])
    delta_ppl = precise_ppl - spec["paper_ppl"]
    try:
        checkpoint = os.path.relpath(checkpoint_path(run_dir), Path.cwd())
    except FileNotFoundError:
        # Statistical extraction can run on a machine that only received the
        # logs and per-sample NLL, not the multi-GB checkpoint.
        checkpoint = "n/a"

    return (
        "=== LM RUN ===\n"
        f"regime: {args['regime']}\n"
        f"config: {config}\n"
        f"hybrid_text_tokens: {args.get('hybrid_text_tokens', 0)}\n"
        f"train_encoder: {str(bool(args.get('train_encoder'))).lower()}\n"
        f"init: {infer_init(args)}\n"
        f"effective_batch: {effective_batch}\n"
        f"tokens_seen_by_decoder: {tokens_seen}\n"
        f"compression_ratio: {compression_ratio:.2f}x\n"
        "\n"
        "=== VAL FINAL ===\n"
        f"val_loss: {metrics['final_val_loss']:.4f}\n"
        f"val_perplexity: {metrics['final_val_perplexity']:.2f}\n"
        f"checkpoint: {checkpoint}\n"
        "\n"
        "=== VS PAPER ===\n"
        f"paper_config: {spec['paper_config']}\n"
        f"paper_ppl: {spec['paper_ppl']:.2f}\n"
        f"delta_ppl: {delta_ppl:+.2f}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Directory containing the run output directories (default: outputs)",
    )
    parser.add_argument(
        "--conv-run",
        type=Path,
        default=None,
        help="Conv1d t=63 run directory (default: <outputs-root>/conv1d_t63_lm_recon-init)",
    )
    parser.add_argument(
        "--vision-run",
        type=Path,
        default=None,
        help="Vision tiny run directory (default: <outputs-root>/vision_tiny_frozen_lm)",
    )
    args = parser.parse_args()

    runs = {
        "conv1d_residual": args.conv_run
        or args.outputs_root / DEFAULT_RUNS["conv1d_residual"]["dirname"],
        "vision": args.vision_run
        or args.outputs_root / DEFAULT_RUNS["vision"]["dirname"],
    }

    missing = False
    for regime, run_dir in runs.items():
        spec = DEFAULT_RUNS[regime]
        try:
            print(format_run(run_dir, spec))
        except (FileNotFoundError, ValueError) as exc:
            missing = True
            print(f"WARNING: skipping {regime} run: {exc}\n")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
