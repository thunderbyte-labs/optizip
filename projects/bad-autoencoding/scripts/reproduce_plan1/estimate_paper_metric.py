#!/usr/bin/env python3
"""Estimate the paper's validation metric with bias-corrected uncertainty.

Primary metric
--------------
The paper reports corpus perplexity:

    PPL = exp(mean_i NLL_i)

where NLL_i is the mean negative log-likelihood of sample i (all samples have
the same number of target tokens in this dataset). The script also reports
mean/median/std of the per-sample NLL and per-sample PPL; these are different
estimands and are labelled explicitly.

Uncertainty
-----------
* bootstrap percentile and BCa confidence intervals;
* bias-corrected point estimates (bootstrap bias for median/PPL, c4 correction
  for the standard deviation, ddof=1 variance);
* Monte-Carlo standard error (MCSE) of the CI bounds;
* a tolerance table for +/-0.01, 0.1, 1, 2, 5, 10, 25, 50, 75, 100:
  coverage, 95% CI half-width, required sample size;
* cumulative-PPL stability and an optional learning-curve projection.

Example
-------
    uv run python scripts/reproduce_plan1/estimate_paper_metric.py \
        --run outputs/conv1d_t63_lm_recon-init \
        --paper-ppl 5.14 \
        --report-dir outputs/report
"""

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

RE_VAL = re.compile(r"Validation loss: ([\d.]+), perplexity: ([\d.]+)")
RE_STEP = re.compile(r"Global: (\d+)\)")
RE_LOADED = re.compile(r"Loaded ([\d,]+) samples from")
RE_CONTEXT = re.compile(r"Effective context tokens \(per-sample\): (\d+)")

TOLERANCES = [0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]


@dataclass
class Metric:
    name: str
    scale: str
    point: float
    boot: np.ndarray
    ci_low: float
    ci_high: float
    ci_method: str = "percentile"
    bias: Optional[float] = None
    corrected: Optional[float] = None


def c4(n: int) -> float:
    """Unbiased standard-deviation correction for a normal sample."""
    if n <= 1:
        return 1.0
    return math.sqrt(2.0 / (n - 1)) * math.exp(
        math.lgamma(n / 2.0) - math.lgamma((n - 1) / 2.0)
    )


def bootstrap_indices(n: int, n_boot: int, seed: int, chunk: int = 500):
    rng = np.random.default_rng(seed)
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        yield rng.integers(0, n, size=(stop - start, n))


def metrics_from_nll(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "corpus_ppl": float(np.exp(x.mean())),
        "mean_nll": float(x.mean()),
        "median_nll": float(np.median(x)),
        "std_nll": float(x.std(ddof=1) / c4(len(x))),
        "mean_ppl": float(np.exp(x).mean()),
        "median_ppl": float(np.exp(np.median(x))),
        "std_ppl": float(np.exp(x).std(ddof=1) / c4(len(x))),
    }


def bootstrap_metrics(x: np.ndarray, n_boot: int, seed: int) -> dict[str, np.ndarray]:
    keys = list(metrics_from_nll(x).keys())
    out = {key: np.empty(n_boot, dtype=np.float64) for key in keys}
    cursor = 0
    for idx in bootstrap_indices(len(x), n_boot, seed):
        sampled = x[idx]
        values = {
            "corpus_ppl": np.exp(sampled.mean(axis=1)),
            "mean_nll": sampled.mean(axis=1),
            "median_nll": np.median(sampled, axis=1),
            "std_nll": sampled.std(axis=1, ddof=1) / c4(x.shape[0]),
            "mean_ppl": np.exp(sampled).mean(axis=1),
            "median_ppl": np.exp(np.median(sampled, axis=1)),
            "std_ppl": np.exp(sampled).std(axis=1, ddof=1) / c4(x.shape[0]),
        }
        stop = cursor + len(idx)
        for key in keys:
            out[key][cursor:stop] = values[key]
        cursor = stop
    return out


def jackknife_acceleration(x: np.ndarray, metric: str) -> float:
    """BCa acceleration from leave-one-out values (full jackknife)."""
    n = len(x)
    values = np.empty(n, dtype=np.float64)
    exp_x = np.exp(x)
    for i in range(n):
        if metric == "mean_nll":
            values[i] = (x.sum() - x[i]) / (n - 1)
        elif metric == "median_nll":
            values[i] = np.median(np.delete(x, i))
        elif metric == "std_nll":
            values[i] = np.delete(x, i).std(ddof=1) / c4(n - 1)
        elif metric == "corpus_ppl":
            values[i] = np.exp((x.sum() - x[i]) / (n - 1))
        elif metric == "mean_ppl":
            values[i] = (exp_x.sum() - exp_x[i]) / (n - 1)
        elif metric == "median_ppl":
            values[i] = np.exp(np.median(np.delete(x, i)))
        elif metric == "std_ppl":
            values[i] = np.delete(exp_x, i).std(ddof=1) / c4(n - 1)
        else:
            raise ValueError(metric)
    mean_jack = values.mean()
    diff = mean_jack - values
    denom = 6.0 * (np.sum(diff**2) ** 1.5)
    return float(np.sum(diff**3) / denom) if denom else 0.0


def bca_interval(
    point: float, boot: np.ndarray, acceleration: float, confidence: float
) -> tuple[float, float]:
    alpha = 1.0 - confidence
    norm = statistics.NormalDist()
    prop = float(np.mean(boot < point))
    prop = min(max(prop, 1.0 / (len(boot) + 1)), 1.0 - 1.0 / (len(boot) + 1))
    z0 = norm.inv_cdf(prop)
    z_alpha = norm.inv_cdf(alpha / 2)
    z_1alpha = norm.inv_cdf(1 - alpha / 2)

    def adjust(z):
        return norm.cdf(z0 + (z0 + z) / (1 - acceleration * (z0 + z)))

    lo = float(np.percentile(boot, 100 * adjust(z_alpha)))
    hi = float(np.percentile(boot, 100 * adjust(z_1alpha)))
    return lo, hi


def mcse_bounds(boot: np.ndarray, confidence: float, n_batches: int = 10) -> float:
    alpha = 1.0 - confidence
    chunks = np.array_split(boot, n_batches)
    lows, highs = [], []
    for chunk in chunks:
        lows.append(np.percentile(chunk, 100 * alpha / 2))
        highs.append(np.percentile(chunk, 100 * (1 - alpha / 2)))
    return float(
        max(np.std(lows, ddof=1), np.std(highs, ddof=1)) / math.sqrt(n_batches)
    )


def build_metrics(
    x: np.ndarray, n_boot: int, seed: int, confidence: float
) -> dict[str, Metric]:
    point = metrics_from_nll(x)
    boot = bootstrap_metrics(x, n_boot, seed)
    out = {}
    for key, value in point.items():
        samples = boot[key]
        lo, hi = np.percentile(samples, [100 * (1 - confidence) / 2,
                                         100 * (1 + confidence) / 2])
        bias = float(samples.mean() - value)
        # The mean (and exp(mean)) has zero jackknife acceleration.
        if key in {"mean_nll", "corpus_ppl", "mean_ppl"}:
            accelerator = 0.0
        else:
            accelerator = jackknife_acceleration(x, key)
        bca_lo, bca_hi = bca_interval(value, samples, accelerator, confidence)
        out[key] = Metric(
            name=key,
            scale="ppl" if "ppl" in key else "nll",
            point=float(value),
            boot=samples,
            ci_low=float(bca_lo),
            ci_high=float(bca_hi),
            ci_method="BCa",
            bias=bias,
            corrected=float(value - bias),
        )
    return out


def tolerance_rows(metrics: dict[str, Metric], tolerances: list[float]) -> list[dict]:
    rows = []
    for metric in metrics.values():
        center = metric.point
        halfwidth = (metric.ci_high - metric.ci_low) / 2
        for eps in tolerances:
            coverage = float(
                np.mean(np.abs(metric.boot - center) <= eps)
            )
            rows.append(
                {
                    "metric": metric.name,
                    "scale": metric.scale,
                    "tolerance": eps,
                    "relative_tolerance_pct": 100 * eps / abs(center) if center else None,
                    "coverage": coverage,
                    "ci95_halfwidth": halfwidth,
                    "meets_target": halfwidth <= eps,
                    # Filled in by main(), where the sample count is known.
                    "required_n": None,
                }
            )
    return rows


def cumulative_curve(x: np.ndarray, step: int = 500) -> list[dict]:
    points = list(range(step, len(x) + 1, step))
    if points[-1] != len(x):
        points.append(len(x))
    return [
        {
            "n": n,
            "ppl": float(np.exp(x[:n].mean())),
            "delta_vs_previous": None
            if idx == 0
            else float(np.exp(x[:n].mean()) - np.exp(x[: points[idx - 1]].mean())),
        }
        for idx, n in enumerate(points)
    ]


def fit_projection(
    steps: list[float], ppls: list[float], n_boot: int, seed: int
) -> Optional[dict]:
    if len(steps) < 4:
        return None
    steps_arr = np.asarray(steps, dtype=np.float64)
    ppl_arr = np.asarray(ppls, dtype=np.float64)
    if not np.all(np.isfinite(ppl_arr)) or np.any(ppl_arr <= 0):
        return None
    # The first validation (reconstruction-initialized model) is often orders
    # of magnitude above the fine-tuned plateau. It dominates a saturating fit,
    # so keep only points within a robust factor of the best observed PPL.
    cutoff = 3.0 * float(ppl_arr.min())
    keep = ppl_arr <= cutoff
    steps_arr = steps_arr[keep]
    ppl_arr = ppl_arr[keep]
    if len(steps_arr) < 4:
        return None
    min_ppl = float(ppl_arr.min())
    # b is a rate per step. The decay happens over hundreds to thousands of
    # steps, so search 1e-6 .. 0.3, not a span-normalised grid.
    b_grid = np.logspace(-6, -0.5, 500)

    def fit(x, y):
        best = None
        for b in b_grid:
            design = np.column_stack([np.ones_like(x), np.exp(-b * x)])
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            c = float(coef[0])
            if not (0.9 * min_ppl <= c <= 1.2 * min_ppl):
                continue
            pred = design @ coef
            sse = float(np.sum((y - pred) ** 2))
            if best is None or sse < best[0]:
                best = (sse, b, coef)
        if best is None:
            return None
        sse, b, coef = best
        return {"c": float(coef[0]), "a": float(coef[1]), "b": float(b), "sse": sse}

    fit_result = fit(steps_arr, ppl_arr)
    if fit_result is None:
        return None
    residuals = ppl_arr - (
        fit_result["c"] + fit_result["a"] * np.exp(-fit_result["b"] * steps_arr)
    )
    rng = np.random.default_rng(seed)
    finals = []
    for _ in range(min(n_boot, 2000)):
        sampled_residuals = rng.choice(residuals, size=len(residuals), replace=True)
        boot_fit = fit(steps_arr, (
            fit_result["c"]
            + fit_result["a"] * np.exp(-fit_result["b"] * steps_arr)
            + sampled_residuals
        ))
        if boot_fit is not None:
            finals.append(boot_fit["c"])
    if len(finals) < 100:
        return None
    finals = np.asarray(finals)
    if not np.all(np.isfinite(finals)) or fit_result["c"] <= 0:
        return None
    lo, hi = np.percentile(finals, [2.5, 97.5])
    return {
        "model": "PPL(t) = c + a * exp(-b * t)",
        "final_ppl_projection": fit_result["c"],
        "ci95": [float(lo), float(hi)],
        "n_eval_points": len(steps_arr),
        "note": "Extrapolation from the validation curve; not a substitute for the final run.",
    }


def load_run(run_dir: Path) -> tuple[np.ndarray, dict, list[tuple[int, float, float]]]:
    per_sample = run_dir / "val_per_sample.jsonl"
    if not per_sample.exists():
        raise SystemExit(
            f"{per_sample} not found. Rerun the final validation with --save_val_per_sample."
        )
    nll = []
    with open(per_sample) as f:
        for line in f:
            if line.strip():
                nll.append(float(json.loads(line)["nll"]))
    x = np.asarray(nll, dtype=np.float64)

    args = {}
    args_path = run_dir / "args.json"
    if args_path.exists():
        with open(args_path) as f:
            args = json.load(f)

    # Parse validation curve with the most recent Global step.
    curve = []
    log_path = run_dir / "train.log"
    last_step = 0
    if log_path.exists():
        for line in log_path.read_text(errors="replace").splitlines():
            step_match = RE_STEP.search(line)
            if step_match:
                last_step = int(step_match.group(1))
            val_match = RE_VAL.search(line)
            if val_match:
                curve.append((last_step, float(val_match.group(1)), float(val_match.group(2))))
    return x, args, curve


def plot_outputs(
    metrics: dict[str, Metric],
    tolerance: list[dict],
    cumulative: list[dict],
    projection: Optional[dict],
    out_dir: Path,
    prefix: str,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    paths = []
    corpus = metrics["corpus_ppl"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(corpus.boot, bins=60, color="#2E5EAA", alpha=0.75)
    ax.axvline(corpus.point, color="black", linewidth=2)
    ax.axvline(corpus.ci_low, color="black", linestyle="--")
    ax.axvline(corpus.ci_high, color="black", linestyle="--")
    ax.set_xlabel("Corpus PPL")
    ax.set_ylabel("Bootstrap resamples")
    ax.set_title("Bootstrap distribution of corpus PPL")
    fig.tight_layout()
    path = out_dir / f"{prefix}_bootstrap_ppl.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path.name)

    cov = [row for row in tolerance if row["metric"] == "corpus_ppl"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([r["tolerance"] for r in cov], [r["coverage"] for r in cov],
            marker="o", color="#2E5EAA")
    ax.set_xscale("log")
    ax.set_xlabel("Absolute tolerance (PPL)")
    ax.set_ylabel("Bootstrap coverage")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.set_title("Coverage of fixed-width intervals")
    fig.tight_layout()
    path = out_dir / f"{prefix}_coverage_vs_tolerance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path.name)

    if cumulative:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([p["n"] for p in cumulative], [p["ppl"] for p in cumulative],
                marker="o", markersize=3, color="#D97706")
        ax.set_xlabel("Validation samples")
        ax.set_ylabel("Cumulative PPL")
        ax.grid(alpha=0.25)
        ax.set_title("Cumulative validation PPL")
        fig.tight_layout()
        path = out_dir / f"{prefix}_cumulative_ppl.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path.name)

    if projection:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        value = projection["final_ppl_projection"]
        ax.bar(["projected final"], [value], color="#2E5EAA")
        ax.errorbar(
            0, value,
            yerr=[[value - projection["ci95"][0]],
                  [projection["ci95"][1] - value]],
            fmt="none", ecolor="black", capsize=6,
        )
        ax.set_ylabel("PPL")
        ax.set_title("Learning-curve projection")
        fig.tight_layout()
        path = out_dir / f"{prefix}_projection.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path.name)
    return paths


def write_report(
    args,
    metrics: dict[str, Metric],
    tolerance: list[dict],
    cumulative: list[dict],
    convergence: dict,
    projection: Optional[dict],
    out_dir: Path,
    prefix: str,
    figures: list[str],
) -> None:
    corpus = metrics["corpus_ppl"]
    n = len(corpus.boot)
    lines = [
        f"# Paper-metric estimate — {args.label or prefix}",
        "",
        f"Run: `{args.run}`",
        f"Validation samples: **{args.n_samples:,}**",
        f"Paper PPL: **{args.paper_ppl:.2f}**" if args.paper_ppl else "Paper PPL: n/a",
        "",
        "## Primary result",
        "",
        "| Metric | Point | Bias-corrected | 95% BCa CI |",
        "|---|---:|---:|---|",
    ]
    for key in ("corpus_ppl", "mean_nll", "median_nll", "std_nll",
                "mean_ppl", "median_ppl", "std_ppl"):
        m = metrics[key]
        lines.append(
            f"| `{key}` | {m.point:.5f} | {m.corrected:.5f} | "
            f"[{m.ci_low:.5f}, {m.ci_high:.5f}] |"
        )
    if args.paper_ppl:
        delta = corpus.point - args.paper_ppl
        inside = corpus.ci_low <= args.paper_ppl <= corpus.ci_high
        lines += [
            "",
            f"- ΔPPL vs paper: **{delta:+.4f}**",
            f"- Paper value is **{'inside' if inside else 'outside'}** the 95% CI.",
        ]
    lines += [
        "",
        "`corpus_ppl` is the paper metric (`exp(mean NLL)`). `mean_ppl` is the "
        "mean of per-sample PPL and is a different estimand (Jensen).",
        "",
        "## Tolerance table (corpus PPL and per-sample metrics)",
        "",
        "| Metric | Tolerance | Coverage | 95% CI half-width | Required n | Meets target |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in tolerance:
        lines.append(
            f"| `{row['metric']}` | ±{row['tolerance']:g} | {row['coverage']:.4f} | "
            f"{row['ci95_halfwidth']:.5f} | {row['required_n']:,} | "
            f"{'yes' if row['meets_target'] else 'no'} |"
        )
    lines += [
        "",
        "## Convergence",
        "",
        f"- MCSE of the 95% CI bounds: **{convergence['mcse']:.6f}** "
        f"(threshold {convergence['mcse_threshold']})",
        f"- Last cumulative PPL change over {args.cumulative_window} samples: "
        f"**{convergence['last_delta']:+.6f}** "
        f"(fixed {convergence['cumulative_threshold_fixed']}, "
        f"effective {convergence['cumulative_threshold']:.6f}; "
        f"SE_diff={convergence['se_diff']:.6f})"
        if convergence.get("se_diff") is not None
        else f"- Last cumulative PPL change: **{convergence['last_delta']:+.6f}**",
        f"- 95% CI half-width: **{convergence['ci_halfwidth']:.5f}** "
        f"(target ≤ {convergence['target_halfwidth']})",
        f"- Converged: **{'yes' if convergence['converged'] else 'no'}**",
        "",
        "Recommended criterion: CI half-width ≤ target, MCSE ≤ threshold, and "
        "cumulative PPL stable over the last window.",
        "",
    ]
    if projection:
        lines += [
            "## Learning-curve projection",
            "",
            f"- Model: `{projection['model']}`",
            f"- Projected final PPL: **{projection['final_ppl_projection']:.4f}** "
            f"(95% CI [{projection['ci95'][0]:.4f}, {projection['ci95'][1]:.4f}])",
            f"- Based on {projection['n_eval_points']} validation points. "
            f"{projection['note']}",
            "",
        ]
    if figures:
        lines += ["## Figures", ""]
        for name in figures:
            lines.append(f"![{name}]({name})")
        lines.append("")
    (out_dir / f"{prefix}_REPORT.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--paper-ppl", type=float, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--tolerances", nargs="+", type=float, default=TOLERANCES)
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-halfwidth", type=float, default=0.05)
    parser.add_argument("--mcse-threshold", type=float, default=0.005)
    parser.add_argument("--cumulative-window", type=int, default=2000)
    parser.add_argument("--cumulative-threshold", type=float, default=0.005)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    x, run_args, curve = load_run(args.run)
    args.n_samples = len(x)
    if args.paper_ppl is None:
        args.paper_ppl = 5.14 if run_args.get("regime") == "conv1d_residual" else 5.35
    if args.report_dir is None:
        args.report_dir = args.run / "estimate"
    args.report_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.label or args.run.name

    metrics = build_metrics(x, args.n_boot, args.seed, args.confidence)
    tolerance = tolerance_rows(metrics, args.tolerances)
    for row in tolerance:
        row["required_n"] = int(
            math.ceil(len(x) * (row["ci95_halfwidth"] / row["tolerance"]) ** 2)
        ) if row["tolerance"] > 0 else None

    cumulative = cumulative_curve(x)
    last_delta = cumulative[-1]["delta_vs_previous"] if cumulative else 0.0
    # Compare the last window with the immediately preceding window of the same
    # size. Comparing with all preceding samples would be dominated by the
    # high-loss initial validation.
    w = args.cumulative_window
    if len(x) > 2 * w:
        window_delta = float(
            np.exp(x[-w:].mean()) - np.exp(x[-2 * w:-w].mean())
        )
    elif len(x) > w:
        window_delta = float(
            np.exp(x[-w:].mean()) - np.exp(x[:-w].mean())
        )
    else:
        window_delta = float(last_delta)

    # Window-to-window noise: the difference of two independent window means
    # has SE = s * sqrt(1/w + 1/w). Use a 95% two-sided bound as the effective
    # stability threshold, with the user's fixed threshold as a floor.
    if len(x) > w:
        se_diff = float(x.std(ddof=1) * math.sqrt(2.0 / w))
        effective_cumulative_threshold = max(
            args.cumulative_threshold, 1.96 * se_diff
        )
    else:
        se_diff = None
        effective_cumulative_threshold = args.cumulative_threshold
    corpus = metrics["corpus_ppl"]
    ci_halfwidth = (corpus.ci_high - corpus.ci_low) / 2
    mcse = mcse_bounds(corpus.boot, args.confidence)
    convergence = {
        "ci_halfwidth": float(ci_halfwidth),
        "target_halfwidth": args.target_halfwidth,
        "mcse": mcse,
        "mcse_threshold": args.mcse_threshold,
        "last_delta": window_delta,
        "cumulative_threshold": effective_cumulative_threshold,
        "cumulative_threshold_fixed": args.cumulative_threshold,
        "se_diff": se_diff,
        "converged": bool(
            ci_halfwidth <= args.target_halfwidth
            and mcse <= args.mcse_threshold
            and abs(window_delta) <= effective_cumulative_threshold
        ),
    }

    projection = None
    if args.paper_ppl is not None and len(curve) >= 3:
        steps = [point[0] for point in curve]
        ppls = [point[2] for point in curve]
        projection = fit_projection(steps, ppls, min(args.n_boot, 2000), args.seed)

    figures = []
    if not args.no_plots:
        figures = plot_outputs(
            metrics, tolerance, cumulative, projection, args.report_dir, prefix
        )

    estimator_rows = []
    for m in metrics.values():
        estimator_rows.append(
            {
                "metric": m.name,
                "scale": m.scale,
                "point": m.point,
                "bias": m.bias,
                "bias_corrected": m.corrected,
                "ci_low": m.ci_low,
                "ci_high": m.ci_high,
                "ci_method": m.ci_method,
                "ci_halfwidth": (m.ci_high - m.ci_low) / 2,
                "mcse_bounds": mcse_bounds(m.boot, args.confidence),
            }
        )
    with open(args.report_dir / f"{prefix}_estimators.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(estimator_rows[0].keys()))
        writer.writeheader()
        writer.writerows(estimator_rows)
    with open(args.report_dir / f"{prefix}_tolerance.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(tolerance[0].keys()))
        writer.writeheader()
        writer.writerows(tolerance)
    with open(args.report_dir / f"{prefix}_estimate.json", "w") as f:
        json.dump(
            {
                "run": str(args.run),
                "n_samples": args.n_samples,
                "paper_ppl": args.paper_ppl,
                "estimators": estimator_rows,
                "tolerance": tolerance,
                "cumulative": cumulative,
                "convergence": convergence,
                "projection": projection,
            },
            f,
            indent=2,
        )
    write_report(
        args, metrics, tolerance, cumulative, convergence, projection,
        args.report_dir, prefix, figures,
    )

    print(f"Wrote {args.report_dir / f'{prefix}_REPORT.md'}")
    print(
        f"corpus PPL: {corpus.point:.5f} "
        f"(95% BCa [{corpus.ci_low:.5f}, {corpus.ci_high:.5f}])"
    )
    print(
        f"convergence: {'PASS' if convergence['converged'] else 'FAIL'} "
        f"(halfwidth={ci_halfwidth:.5f}, mcse={mcse:.5f}, delta={window_delta:+.5f})"
    )
    if projection:
        print(
            f"projected final PPL: {projection['final_ppl_projection']:.4f} "
            f"(95% CI [{projection['ci95'][0]:.4f}, {projection['ci95'][1]:.4f}])"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
