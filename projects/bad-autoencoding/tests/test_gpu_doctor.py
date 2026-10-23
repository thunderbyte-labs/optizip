"""Tests for gpu_doctor parsers and verdicts."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reproduce_plan1"
    / "gpu_doctor.py"
)
spec = importlib.util.spec_from_file_location("gpu_doctor", MODULE_PATH)
gpu_doctor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_doctor)


def test_unit_conversions():
    assert gpu_doctor._mib_to_gb("1024") == 1.0
    assert gpu_doctor._bytes_to_gb(str(1024**3)) == 1.0
    assert gpu_doctor._num("N/A") is None
    assert gpu_doctor._num("95 %") == 95.0


def test_igpu_detection():
    assert gpu_doctor._is_igpu({"name": "AMD Radeon Graphics", "mem_total_gb": 2.0})
    assert not gpu_doctor._is_igpu(
        {"name": "AMD Radeon RX 9070 XT", "mem_total_gb": 15.9}
    )


def test_feasibility_pass_and_fail(tmp_path, capsys):
    result = {
        "passed": True,
        "cpu_offload": False,
        "optimizer": "adamw8bit",
        "world_size": 2,
        "memory_limit_gb": 14.3,
        "per_rank": [
            {"rank": 0, "peak_allocated_gb": 13.0, "peak_reserved_gb": 13.5,
             "headroom_gb": 2.9, "total_gb": 15.9},
            {"rank": 1, "peak_allocated_gb": 13.2, "peak_reserved_gb": 13.7,
             "headroom_gb": 2.7, "total_gb": 15.9},
        ],
    }
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(result))
    args = argparse.Namespace(
        probe_json=str(path), limit_gb=0.0, min_headroom_gb=0.5,
        require_no_cpu_offload=True, json=False,
    )
    assert gpu_doctor.cmd_feasibility(args) == 0
    assert "PASS" in capsys.readouterr().out

    result["per_rank"][0]["peak_allocated_gb"] = 15.5
    path.write_text(json.dumps(result))
    assert gpu_doctor.cmd_feasibility(args) == 1

    result["passed"] = True
    result["per_rank"][0]["peak_allocated_gb"] = 13.0
    result["cpu_offload"] = True
    path.write_text(json.dumps(result))
    assert gpu_doctor.cmd_feasibility(args) == 1


def test_report_marks_healthy_monitor_as_pass(tmp_path, capsys):
    csv_path = tmp_path / "monitor.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=gpu_doctor.GPU_FIELDS)
        writer.writeheader()
        for util in (95.0, 96.0, 94.0):
            writer.writerow(
                {
                    "timestamp": "2026-01-01T00:00:00",
                    "index": 0,
                    "name": "GPU0",
                    "util_gpu": util,
                    "util_mem": 50,
                    "mem_used_gb": 10.0,
                    "mem_total_gb": 16.0,
                    "power_w": 200,
                    "temp_c": 60,
                    "clock_mhz": 2000,
                    "throttle": None,
                }
            )
    out = tmp_path / "report.json"
    args = argparse.Namespace(
        monitor_csv=str(csv_path), train_log=None, probe_json=None,
        util_threshold=85.0, max_idle_fraction=0.10,
        max_throughput_cv=5.0, out=str(out), json=False,
    )
    assert gpu_doctor.cmd_report(args) == 0
    payload = json.loads(out.read_text())
    assert payload["overall_pass"] is True
    assert payload["checks"]["util_p50_ge_threshold"] is True
