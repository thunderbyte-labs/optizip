#!/usr/bin/env python3
"""Build a statistical report comparing PLAN_1 runs with the paper.

The script reads the run directories produced by 02_conv1d_t63.sh and
03_vision_tiny_frozen.sh:

    <run>/train.log
    <run>/args.json
    <run>/val_per_sample.jsonl   (written by --save_val_per_sample)

It produces:

    <report-dir>/REPORT.md
    <report-dir>/summary.json
    <report-dir>/fig1_ppl_vs_paper.png
    <report-dir>/fig2_delta_vs_paper.png
    <report-dir>/fig3_bootstrap_ppl.png
    <report-dir>/fig4_paired_delta.png
    <report-dir>/fig5_validation_curves.png
    <report-dir>/fig6_per_sample_ecdf.png

Examples:
    uv run python scripts/reproduce_plan1/make_report.py
    uv run python scripts/reproduce_plan1/make_report.py \
        --conv-run outputs/conv1d_t63_lm_recon-init \
        --vision-run outputs/vision_tiny_frozen_lm
"""

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RE_VAL = re.compile(r"Validation loss: ([\d.]+), perplexity: ([\d.]+)")
RE_CONTEXT = re.compile(r"Effective context tokens \(per-sample\): (\d+)")
RE_TARGET = re.compile(r"Target tokens per sample: (\d+)")
RE_LOADED = re.compile(r"Loaded ([\d,]+) samples from")

CONV_COLOR = "#2E5EAA"
VISION_COLOR = "#D97706"
PAPER_COLOR = "#6B7280"

DEFAULT_RUNS = {
    "conv1d_residual": {
        "dirname": "conv1d_t63_lm_recon-init",
        "label": "Hierarchical t=63",
        "short": "conv1d t=63",
        "paper_config": "Hierarchical t=63 (Recon./Recon.)",
        "paper_ppl": 5.14,
        "paper_tokens": 65,
        "color": CONV_COLOR,
    },
    "vision": {
        "dirname": "vision_tiny_frozen_lm",
        "label": "Vision tiny frozen",
        "short": "vision tiny frozen",
        "paper_config": "Vision tiny frozen (DS frozen/DS)",
        "paper_ppl": 5.35,
        "paper_tokens": 78,
        "color": VISION_COLOR,
    },
}


@dataclass
class RunData:
    regime: str
    run_dir: Path
    args: dict = field(default_factory=dict)
    num_samples: Optional[int] = None
    context_tokens: Optional[int] = None
    target_tokens: Optional[int] = None
    val_loss: Optional[float] = None
    val_ppl: Optional[float] = None
    nll: Optional[np.ndarray] = None
    sample_ids: Optional[list] = None
    validation_curve: list = field(default_factory=list)
    status: str = "missing"
    error: Optional[str] = None

    @property
    def has_per_sample(self) -> bool:
        return self.nll is not None and len(self.nll) > 0

    @property
    def ppl(self) -> Optional[float]:
        if self.has_per_sample:
            return float(math.exp(float(self.nll.mean())))
        return self.val_ppl

    @property
    def n(self) -> int:
        return len(self.nll) if self.has_per_sample else 0


def read_run(regime: str, run_dir: Path) -> RunData:
    run = RunData(regime=regime, run_dir=run_dir)
    args_path = run_dir / "args.json"
    log_path = run_dir / "train.log"

    if args_path.exists():
        with open(args_path) as f:
            run.args = json.load(f)
    if log_path.exists():
        text = log_path.read_text(errors="replace")
        matches = RE_VAL.findall(text)
        if matches:
            run.val_loss, run.val_ppl = (float(x) for x in matches[-1])
            run.validation_curve = [(float(loss), float(ppl)) for loss, ppl in matches]
        loaded = RE_LOADED.search(text)
        context = RE_CONTEXT.search(text)
        target = RE_TARGET.search(text)
        if loaded:
            run.num_samples = int(loaded.group(1).replace(",", ""))
        if context:
            run.context_tokens = int(context.group(1))
        if target:
            run.target_tokens = int(target.group(1))

    per_sample_path = run_dir / "val_per_sample.jsonl"
    if per_sample_path.exists():
        ids, nlls = [], []
        with open(per_sample_path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                ids.append(record["sample_id"])
                nlls.append(float(record["nll"]))
        if nlls:
            run.sample_ids = ids
            run.nll = np.asarray(nlls, dtype=np.float64)

    if run.has_per_sample:
        run.status = "complete"
    elif run.val_loss is not None:
        run.status = "complete (aggregate only)"
    elif args_path.exists() or log_path.exists():
        run.status = "incomplete"
    if not run_dir.exists():
        run.error = f"run directory not found: {run_dir}"
    return run


def bootstrap_means(x: np.ndarray, n_boot: int = 5000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(x)
    out = np.empty(n_boot, dtype=np.float64)
    chunk = 500
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        idx = rng.integers(0, n, size=(stop - start, n))
        out[start:stop] = x[idx].mean(axis=1)
    return out


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(a)
    diff_nll = np.empty(n_boot, dtype=np.float64)
    diff_ppl = np.empty(n_boot, dtype=np.float64)
    chunk = 500
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        idx = rng.integers(0, n, size=(stop - start, n))
        a_mean = a[idx].mean(axis=1)
        b_mean = b[idx].mean(axis=1)
        diff_nll[start:stop] = a_mean - b_mean
        diff_ppl[start:stop] = np.exp(a_mean) - np.exp(b_mean)
    return diff_nll, diff_ppl


def paired_permutation_pvalue(diff: np.ndarray, n_perm: int = 10000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    observed = float(np.mean(diff))
    count = 0
    chunk = 1000
    for start in range(0, n_perm, chunk):
        stop = min(start + chunk, n_perm)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(stop - start, len(diff)))
        perm_means = (signs * diff).mean(axis=1)
        count += int(np.sum(np.abs(perm_means) >= abs(observed)))
    return (count + 1) / (n_perm + 1)


def ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def fmt_ppl(value: Optional[float], digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_delta(value: float) -> str:
    return f"{value:+.3f}"


def significance_text(p_value: float, lo: float, hi: float) -> str:
    if p_value < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {p_value:.3f}"
    excludes_zero = lo > 0 or hi < 0
    if excludes_zero and p_value < 0.05:
        return f"difference statistically significant ({p_text}, 95% CI excludes 0)"
    if not excludes_zero and p_value >= 0.05:
        return f"no statistically significant difference ({p_text}, 95% CI includes 0)"
    return f"inconclusive ({p_text}; bootstrap CI and permutation test disagree)"


def load_nll(run: RunData) -> Optional[np.ndarray]:
    return run.nll if run.has_per_sample else None


def plot_ppl_vs_paper(runs: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(runs))
    width = 0.34
    ours, ours_err, paper = [], [], []
    labels = []
    for run in runs.values():
        spec = DEFAULT_RUNS[run.regime]
        labels.append(spec["short"])
        paper.append(spec["paper_ppl"])
        if run.ppl is None:
            ours.append(np.nan)
            ours_err.append(0.0)
        else:
            ours.append(run.ppl)
            if run.has_per_sample:
                boot = np.exp(bootstrap_means(run.nll))
                lo, hi = ci(boot)
                ours_err.append((hi - lo) / 2)
            else:
                ours_err.append(0.0)
    ax.bar(x - width / 2, ours, width, yerr=ours_err, capsize=4,
           color=[DEFAULT_RUNS[r.regime]["color"] for r in runs.values()],
           label="This reproduction (95% bootstrap CI)")
    ax.bar(x + width / 2, paper, width, color=PAPER_COLOR, alpha=0.55,
           label="Paper (point estimate)")
    for i, (o, p) in enumerate(zip(ours, paper)):
        if np.isfinite(o):
            ax.text(i - width / 2, o + 0.03, f"{o:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, p + 0.03, f"{p:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Validation perplexity")
    ax.set_title("Continuation perplexity: reproduction vs paper")
    ax.set_ylim(0, max([v for v in ours + paper if np.isfinite(v)] + [6.0]) + 0.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_delta_vs_paper(runs: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels, deltas, errs, colors = [], [], [], []
    for run in runs.values():
        spec = DEFAULT_RUNS[run.regime]
        labels.append(spec["short"])
        colors.append(spec["color"])
        if run.ppl is None:
            deltas.append(np.nan)
            errs.append(0.0)
            continue
        deltas.append(run.ppl - spec["paper_ppl"])
        if run.has_per_sample:
            boot_ppl = np.exp(bootstrap_means(run.nll))
            lo, hi = ci(boot_ppl)
            errs.append((hi - lo) / 2)
        else:
            errs.append(0.0)
    y = np.arange(len(labels))
    ax.barh(y, deltas, xerr=errs, capsize=4, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    for i, d in enumerate(deltas):
        if np.isfinite(d):
            ax.text(d + (0.02 if d >= 0 else -0.02), i, fmt_delta(d),
                    va="center", ha="left" if d >= 0 else "right", fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Our PPL - paper PPL")
    ax.set_title("Signed difference from the paper (positive = worse)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bootstrap(runs: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for run in runs.values():
        spec = DEFAULT_RUNS[run.regime]
        if not run.has_per_sample:
            continue
        boot = np.exp(bootstrap_means(run.nll))
        ax.hist(boot, bins=60, alpha=0.55, color=spec["color"],
                label=f"{spec['short']} bootstrap (ours)")
        ax.axvline(run.ppl, color=spec["color"], linewidth=2)
        ax.axvline(spec["paper_ppl"], color=spec["color"], linestyle="--", linewidth=1.5)
        lo, hi = ci(boot)
        ax.axvspan(lo, hi, color=spec["color"], alpha=0.08)
    ax.set_xlabel("Validation perplexity")
    ax.set_ylabel("Bootstrap resamples")
    ax.set_title("Bootstrap uncertainty of our PPL (dashed = paper)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_paired(
    diff_ppl: np.ndarray, diff_nll: np.ndarray, p_value: float, path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    lo, hi = ci(diff_ppl)
    axes[0].hist(diff_ppl, bins=60, color=CONV_COLOR, alpha=0.75)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].axvline(float(np.mean(diff_ppl)), color="black", linestyle="--", linewidth=1.5)
    axes[0].axvspan(lo, hi, color="black", alpha=0.08)
    axes[0].set_title("Paired ΔPPL (conv1d - vision)")
    axes[0].set_xlabel("ΔPPL")
    axes[0].set_ylabel("Bootstrap resamples")
    axes[0].grid(alpha=0.2)
    axes[0].text(0.02, 0.95, f"95% CI [{lo:+.3f}, {hi:+.3f}]\n{p_value=:.4f}",
                 transform=axes[0].transAxes, va="top", fontsize=9)

    axes[1].hist(diff_nll, bins=60, color=VISION_COLOR, alpha=0.75)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].axvline(float(np.mean(diff_nll)), color="black", linestyle="--", linewidth=1.5)
    axes[1].set_title("Paired ΔNLL (conv1d - vision)")
    axes[1].set_xlabel("ΔNLL (negative = conv1d better)")
    axes[1].grid(alpha=0.2)
    fig.suptitle("Same validation samples: paired uncertainty")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_curves(runs: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for run in runs.values():
        if not run.validation_curve:
            continue
        spec = DEFAULT_RUNS[run.regime]
        ppls = [ppl for _, ppl in run.validation_curve]
        ax.plot(range(len(ppls)), ppls, marker="o", markersize=3,
                color=spec["color"], label=spec["short"])
    ax.set_xlabel("Validation pass")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("Validation perplexity during training")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ecdf(runs: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for run in runs.values():
        if not run.has_per_sample:
            continue
        spec = DEFAULT_RUNS[run.regime]
        x = np.sort(run.nll)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=spec["color"], linewidth=2, label=spec["short"])
    ax.set_xlabel("Per-sample mean NLL")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Per-sample validation loss distribution")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_summary(runs: dict, paired: Optional[dict]) -> dict:
    summary = {"runs": {}, "paired": paired}
    for run in runs.values():
        spec = DEFAULT_RUNS[run.regime]
        entry = {
            "regime": run.regime,
            "run_dir": str(run.run_dir),
            "status": run.status,
            "paper_ppl": spec["paper_ppl"],
            "paper_tokens": spec["paper_tokens"],
            "context_tokens": run.context_tokens,
            "num_samples": run.num_samples,
            "effective_batch": int(run.args.get("batch_size", 0))
            * int(run.args.get("gradient_accumulation_steps", 0)),
            "computed_from": "per_sample_nll" if run.has_per_sample else "train_log",
        }
        if run.ppl is not None:
            entry["ppl"] = run.ppl
            entry["delta_ppl"] = run.ppl - spec["paper_ppl"]
        if run.has_per_sample:
            boot = np.exp(bootstrap_means(run.nll))
            lo, hi = ci(boot)
            entry["ppl_ci95"] = [lo, hi]
            entry["mean_nll"] = float(run.nll.mean())
            entry["std_nll"] = float(run.nll.std(ddof=1))
        summary["runs"][run.regime] = entry
    return summary


def write_report(
    runs: dict,
    summary: dict,
    paired: Optional[dict],
    report_dir: Path,
    figures: dict,
) -> None:
    conv = runs["conv1d_residual"]
    vision = runs["vision"]
    conv_spec = DEFAULT_RUNS["conv1d_residual"]
    vision_spec = DEFAULT_RUNS["vision"]

    lines = [
        "# PLAN_1 — Language-modeling reproduction report",
        "",
        f"Generated: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}",
        "",
        "## What was reproduced",
        "",
        "| Run | Regime | Init | Context tokens | Paper PPL | Ours PPL | ΔPPL |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for run in (conv, vision):
        spec = DEFAULT_RUNS[run.regime]
        init = run.args.get("init_from_checkpoint")
        init_label = "reconstruction checkpoint" if init else "DeepSeek-OCR (DS)"
        ours = fmt_ppl(run.ppl)
        delta = fmt_delta(run.ppl - spec["paper_ppl"]) if run.ppl is not None else "n/a"
        lines.append(
            f"| {spec['label']} | `{run.regime}` | {init_label} | "
            f"{run.context_tokens or spec['paper_tokens']} | {spec['paper_ppl']:.2f} | "
            f"{ours} | {delta} |"
        )
    lines += [
        "",
        "The paper reports point estimates only. The uncertainty below comes from "
        "the 10k validation samples, not from retraining with different seeds.",
        "",
        "## Statistical analysis",
        "",
    ]

    for run in (conv, vision):
        spec = DEFAULT_RUNS[run.regime]
        if run.has_per_sample:
            boot = np.exp(bootstrap_means(run.nll))
            lo, hi = ci(boot)
            inside = lo <= spec["paper_ppl"] <= hi
            lines += [
                f"### {spec['label']}",
                "",
                f"- Our PPL: **{run.ppl:.3f}** (95% bootstrap CI [{lo:.3f}, {hi:.3f}], "
                f"n = {run.n:,} samples)",
                f"- Paper PPL: **{spec['paper_ppl']:.2f}**, Δ = **{fmt_delta(run.ppl - spec['paper_ppl'])}**",
                f"- The paper value is **{'inside' if inside else 'outside'}** our 95% CI.",
                "",
            ]
        elif run.ppl is not None:
            lines += [
                f"### {spec['label']}",
                "",
                f"- Our PPL: **{run.ppl:.3f}** (aggregate only; per-sample data unavailable, "
                "so no CI was computed)",
                f"- Paper PPL: **{spec['paper_ppl']:.2f}**, Δ = **{fmt_delta(run.ppl - spec['paper_ppl'])}**",
                "",
            ]
        else:
            lines += [
                f"### {spec['label']}",
                "",
                f"- **Run not available yet.** Expected directory: `{run.run_dir}`.",
                "",
            ]

    if paired is not None:
        lines += [
            "### Paired comparison: hierarchical vs vision",
            "",
            "Both runs were evaluated on the same validation samples, so the paired "
            "analysis removes sample difficulty from the comparison.",
            "",
            f"- Mean per-sample NLL: conv1d **{paired['mean_nll_conv']:.4f}**, "
            f"vision **{paired['mean_nll_vision']:.4f}**",
            f"- Mean paired ΔNLL (conv1d - vision): **{paired['mean_diff_nll']:+.4f}** "
            f"(95% CI [{paired['ci_nll'][0]:+.4f}, {paired['ci_nll'][1]:+.4f}])",
            f"- Mean paired ΔPPL (conv1d - vision): **{paired['mean_diff_ppl']:+.3f}** "
            f"(95% CI [{paired['ci_ppl'][0]:+.3f}, {paired['ci_ppl'][1]:+.3f}])",
            f"- Sign-flip permutation test: **p = {paired['p_value']:.4f}**",
            f"- Samples where conv1d has lower NLL: **{paired['conv_better_frac'] * 100:.1f}%**",
            f"- Paired effect size (Cohen's d_z): **{paired['cohen_dz']:+.3f}**",
            f"- Verdict: {significance_text(paired['p_value'], *paired['ci_ppl'])}.",
            "",
        ]
    else:
        lines += [
            "### Paired comparison: hierarchical vs vision",
            "",
            "Not available: both runs must have `val_per_sample.jsonl`.",
            "",
        ]

    lines += ["## Figures", ""]
    if figures.get("ppl_vs_paper"):
        lines += [
            "### PPL vs paper",
            f"![PPL vs paper]({figures['ppl_vs_paper']})",
            "",
        ]
    if figures.get("delta_vs_paper"):
        lines += [
            "### Signed difference from the paper",
            f"![Delta vs paper]({figures['delta_vs_paper']})",
            "",
        ]
    if figures.get("bootstrap"):
        lines += ["### Bootstrap uncertainty", f"![Bootstrap]({figures['bootstrap']})", ""]
    if figures.get("paired"):
        lines += ["### Paired difference", f"![Paired]({figures['paired']})", ""]
    if figures.get("curves"):
        lines += ["### Validation curves", f"![Curves]({figures['curves']})", ""]
    if figures.get("ecdf"):
        lines += ["### Per-sample loss distribution", f"![ECDF]({figures['ecdf']})", ""]

    lines += [
        "## How to read the significance claims",
        "",
        "1. **Within-run uncertainty.** The bootstrap CI quantifies how precisely the "
        "10k-sample validation set pins down each PPL. It does **not** include "
        "training-seed variance.",
        "2. **Paper comparison.** The paper reports a single point estimate per row and "
        "does not publish per-sample outputs. A paper value outside our CI means the "
        "gap is larger than validation sampling noise, but it is not a formal "
        "two-sample test: hardware, FlashAttention vs eager attention, and possible "
        "data-order differences also contribute.",
        "3. **Method comparison.** The paired test uses the same validation samples for "
        "both methods, so it directly tests the paper's claim that the hierarchical "
        "encoder beats vision at a matched token budget. A negative ΔPPL means the "
        "hierarchical encoder is better.",
        "4. **Scope.** One training run per configuration cannot estimate run-to-run "
        "variance. Repeating the runs with different seeds would be needed for a "
        "fully powered comparison.",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Effective batch size | {summary['runs'].get('conv1d_residual', {}).get('effective_batch', 48)} |",
        "| Learning rate | 1e-4 |",
        "| Epochs | 1 |",
        "| Validation samples | 10,000 |",
        "| Attention backend | eager (no aarch64 FlashAttention-2 wheel) |",
        "",
        "## Reproduce",
        "",
        "```bash",
        "bash scripts/reproduce_plan1/00_setup_gb10.sh",
        "bash scripts/reproduce_plan1/02_conv1d_t63.sh",
        "bash scripts/reproduce_plan1/03_vision_tiny_frozen.sh",
        "uv run python scripts/reproduce_plan1/make_report.py",
        "```",
        "",
        "## Raw files",
        "",
    ]
    for run in (conv, vision):
        lines.append(f"- `{run.run_dir}` — status: **{run.status}**")
    lines.append("")

    (report_dir / "REPORT.md").write_text("\n".join(lines))
    with open(report_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--conv-run", type=Path, default=None)
    parser.add_argument("--vision-run", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report_dir = args.report_dir or args.outputs_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {
        "conv1d_residual": args.conv_run
        or args.outputs_root / DEFAULT_RUNS["conv1d_residual"]["dirname"],
        "vision": args.vision_run
        or args.outputs_root / DEFAULT_RUNS["vision"]["dirname"],
    }
    runs = {regime: read_run(regime, path) for regime, path in run_dirs.items()}

    figures = {}
    if any(run.ppl is not None for run in runs.values()):
        p = report_dir / "fig1_ppl_vs_paper.png"
        plot_ppl_vs_paper(runs, p)
        figures["ppl_vs_paper"] = p.name

        p = report_dir / "fig2_delta_vs_paper.png"
        plot_delta_vs_paper(runs, p)
        figures["delta_vs_paper"] = p.name

    if any(run.has_per_sample for run in runs.values()):
        p = report_dir / "fig3_bootstrap_ppl.png"
        plot_bootstrap(runs, p)
        figures["bootstrap"] = p.name

    if any(run.validation_curve for run in runs.values()):
        p = report_dir / "fig5_validation_curves.png"
        plot_curves(runs, p)
        figures["curves"] = p.name

    if all(run.has_per_sample for run in runs.values()):
        p = report_dir / "fig6_per_sample_ecdf.png"
        plot_ecdf(runs, p)
        figures["ecdf"] = p.name

    paired = None
    if all(run.has_per_sample for run in runs.values()):
        conv = runs["conv1d_residual"]
        vision = runs["vision"]
        common = sorted(set(conv.sample_ids) & set(vision.sample_ids))
        if len(common) < 100:
            print(f"WARNING: only {len(common)} paired samples; skipping paired analysis")
        else:
            conv_map = dict(zip(conv.sample_ids, conv.nll))
            vision_map = dict(zip(vision.sample_ids, vision.nll))
            a = np.array([conv_map[s] for s in common])
            b = np.array([vision_map[s] for s in common])
            diff_nll = a - b
            diff_nll_boot, diff_ppl_boot = paired_bootstrap(a, b, n_boot=args.n_boot, seed=args.seed)
            p_value = paired_permutation_pvalue(diff_nll, n_perm=args.n_perm, seed=args.seed)
            paired = {
                "n_pairs": len(common),
                "mean_nll_conv": float(a.mean()),
                "mean_nll_vision": float(b.mean()),
                "mean_diff_nll": float(diff_nll.mean()),
                "ci_nll": list(ci(diff_nll_boot)),
                "mean_diff_ppl": float(np.exp(a.mean()) - np.exp(b.mean())),
                "ci_ppl": list(ci(diff_ppl_boot)),
                "p_value": float(p_value),
                "conv_better_frac": float(np.mean(diff_nll < 0)),
                "cohen_dz": float(diff_nll.mean() / diff_nll.std(ddof=1))
                if diff_nll.std(ddof=1) > 0 else 0.0,
            }
            p = report_dir / "fig4_paired_delta.png"
            plot_paired(diff_ppl_boot, diff_nll_boot, p_value, p)
            figures["paired"] = p.name

    summary = build_summary(runs, paired)
    write_report(runs, summary, paired, report_dir, figures)

    print(f"Report written to {report_dir / 'REPORT.md'}")
    for run in runs.values():
        print(f"  {run.regime}: {run.status} (PPL={fmt_ppl(run.ppl)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
