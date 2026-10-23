"""Tests for the paper-metric estimator."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reproduce_plan1"
    / "estimate_paper_metric.py"
)
spec = importlib.util.spec_from_file_location("estimate_paper_metric", MODULE_PATH)
estimators = importlib.util.module_from_spec(spec)
spec.loader.exec_module(estimators)


def test_c4_is_close_to_one_for_large_samples():
    assert estimators.c4(10_000) == pytest.approx(0.99995, abs=1e-4)
    assert estimators.c4(1) == 1.0


def test_corpus_ppl_is_exp_mean_nll():
    x = np.log(np.array([2.0, 4.0, 8.0]))
    metrics = estimators.metrics_from_nll(x)
    assert metrics["corpus_ppl"] == pytest.approx(4.0)
    assert metrics["mean_ppl"] > metrics["corpus_ppl"]


def test_bootstrap_has_expected_shape():
    rng = np.random.default_rng(0)
    x = rng.normal(1.6, 0.3, 200)
    boot = estimators.bootstrap_metrics(x, n_boot=300, seed=0)
    assert boot["corpus_ppl"].shape == (300,)
    assert np.all(np.isfinite(boot["median_nll"]))


def test_bca_interval_contains_point_for_well_behaved_sample():
    rng = np.random.default_rng(0)
    x = rng.normal(1.6, 0.3, 500)
    metrics = estimators.build_metrics(x, n_boot=500, seed=0, confidence=0.95)
    m = metrics["corpus_ppl"]
    assert m.ci_low < m.point < m.ci_high
    assert m.ci_low < m.corrected < m.ci_high


def test_projection_recovers_saturating_asymptote():
    steps = list(range(0, 5001, 500))
    ppls = [5.10 + 4.0 * np.exp(-s / 900.0) for s in steps]
    result = estimators.fit_projection(steps, ppls, n_boot=200, seed=0)
    assert result is not None
    assert result["final_ppl_projection"] == pytest.approx(5.10, abs=0.05)
    assert result["ci95"][0] <= result["final_ppl_projection"] <= result["ci95"][1]


def test_tolerance_rows_mark_large_tolerances_as_meeting_target():
    x = np.linspace(1.0, 2.0, 1000)
    metrics = estimators.build_metrics(x, n_boot=200, seed=0, confidence=0.95)
    rows = estimators.tolerance_rows(metrics, [0.01, 10.0])
    corpus = [row for row in rows if row["metric"] == "corpus_ppl"]
    small, large = corpus
    assert large["meets_target"] is True
    assert small["meets_target"] in (True, False)
