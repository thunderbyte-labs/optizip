#!/usr/bin/env python3
"""Cross-vendor GPU diagnostics for the PLAN_1 reproduction.

Subcommands:
    env          inventory of drivers, PyTorch, GPUs and tools
    monitor      sample utilization/memory/power/temperature to CSV
    feasibility  read train_fsdp.py --probe output and print a go/no-go verdict
    report       summarize a monitor CSV (and optional train log/probe)
    dist         measure all-reduce bandwidth across GPUs
    sweep        run short single-GPU configs and report throughput

The tool works with NVIDIA (nvidia-smi) and AMD (rocm-smi / amd-smi). It never
changes model or training code, and it does not perform CPU offload.
"""

import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

GPU_FIELDS = [
    "timestamp",
    "index",
    "name",
    "util_gpu",
    "util_mem",
    "mem_used_gb",
    "mem_total_gb",
    "power_w",
    "temp_c",
    "clock_mhz",
    "throttle",
]


def _run(cmd, timeout=10):
    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _num(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text in {"", "N/A", "n/a", "NA", "[N/A]", "Not Supported"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group(0)) if match else None


def _mib_to_gb(value):
    value = _num(value)
    return value / 1024 if value is not None else None


def _bytes_to_gb(value):
    value = _num(value)
    return value / 1024**3 if value is not None else None


def _tool(name):
    path = shutil.which(name)
    if not path:
        return {"available": False, "path": None, "version": None}
    result = _run([path, "--version"])
    version = (result.stdout.strip().splitlines()[0] if result and result.stdout else None)
    return {"available": True, "path": path, "version": version}


def _torch_info():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}
    info = {
        "available": True,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info["devices"].append(
                {
                    "index": index,
                    "name": props.name,
                    "total_gb": round(props.total_memory / 1024**3, 2),
                    "major": props.major,
                    "minor": props.minor,
                    "gfx_arch": getattr(props, "gcnArchName", None),
                }
            )
    return info


def _smi_nvidia():
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,utilization.memory,"
            "memory.used,memory.total,power.draw,temperature.gpu,"
            "clocks.sm,clocks.max.sm",
            "--format=csv,noheader,nounits",
        ]
    )
    if result is None or result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 10:
            continue
        rows.append(
            {
                "index": int(_num(parts[0]) or len(rows)),
                "name": parts[1],
                "util_gpu": _num(parts[2]),
                "util_mem": _num(parts[3]),
                "mem_used_gb": _mib_to_gb(parts[4]),
                "mem_total_gb": _mib_to_gb(parts[5]),
                "power_w": _num(parts[6]),
                "temp_c": _num(parts[7]),
                "clock_mhz": _num(parts[8]),
                "clock_max_mhz": _num(parts[9]),
                "throttle": None,
            }
        )
    return rows


def _smi_amd():
    """Parse rocm-smi text output. Returns one dict per GPU."""
    result = _run(
        [
            "rocm-smi",
            "--showuse",
            "--showmeminfo",
            "vram",
            "--showtemp",
            "--showpower",
            "--showclocks",
        ]
    )
    if result is None or result.returncode != 0:
        return []

    rows = {}
    for line in result.stdout.splitlines():
        match = re.search(r"GPU\[(\d+)\]\s*:\s*(.*?):\s*(.+)$", line.strip())
        if not match:
            continue
        index = int(match.group(1))
        key = match.group(2).strip().lower()
        value = match.group(3).strip()
        row = rows.setdefault(index, {"index": index, "name": None, "throttle": None})

        if "gpu use" in key:
            row["util_gpu"] = _num(value)
        elif "vram total memory" in key:
            row["mem_total_gb"] = _bytes_to_gb(value)
        elif "vram total used memory" in key:
            row["mem_used_gb"] = _bytes_to_gb(value)
        elif "temperature" in key:
            row["temp_c"] = _num(value)
        elif "power" in key:
            row["power_w"] = _num(value)
        elif "sclk" in key or "clock" in key:
            row["clock_mhz"] = _num(value)
        elif "card series" in key:
            row["name"] = value

    # rocm-smi --showproductname is more reliable for names.
    product = _run(["rocm-smi", "--showproductname"])
    if product and product.returncode == 0:
        current = None
        for line in product.stdout.splitlines():
            index_match = re.search(r"GPU\[(\d+)\]", line)
            if index_match:
                current = int(index_match.group(1))
            name_match = re.search(r"Card Series:\s*(.+)", line)
            if name_match and current is not None and current in rows:
                rows[current]["name"] = name_match.group(1).strip()

    # Memory fallback from PyTorch (rocm-smi may be unavailable in containers).
    try:
        import torch

        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                if index not in rows:
                    continue
                free, total = torch.cuda.mem_get_info(index)
                rows[index].setdefault("mem_total_gb", total / 1024**3)
                rows[index].setdefault("mem_used_gb", (total - free) / 1024**3)
    except Exception:
        pass

    for row in rows.values():
        row.setdefault("util_mem", None)
        row.setdefault("mem_used_gb", None)
        row.setdefault("mem_total_gb", None)
        row.setdefault("power_w", None)
        row.setdefault("temp_c", None)
        row.setdefault("clock_mhz", None)
    return sorted(rows.values(), key=lambda item: item["index"])


def _smi(device=None):
    if shutil.which("nvidia-smi"):
        rows = _smi_nvidia()
    elif shutil.which("rocm-smi"):
        rows = _smi_amd()
    else:
        rows = []
    if device is not None:
        rows = [row for row in rows if row["index"] == device]
    return rows


def _is_igpu(row):
    name = (row.get("name") or "").lower()
    total = _num(row.get("mem_total_gb"))
    return (
        "raphael" in name
        or "graphics" in name
        or (total is not None and total < 4)
    )


def _print_json(data):
    print(json.dumps(data, indent=2, default=str))


def cmd_env(args):
    torch_info = _torch_info()
    smi_rows = _smi()
    tools = {name: _tool(name) for name in ("nvidia-smi", "rocm-smi", "amd-smi", "nvcc", "hipcc")}
    devices = [dict(row) for row in smi_rows]

    # Merge torch devices by index when smi does not expose memory (GB10).
    if torch_info.get("available"):
        for td in torch_info.get("devices", []):
            match = next((d for d in devices if d["index"] == td["index"]), None)
            if match is None:
                devices.append(
                    {
                        "index": td["index"],
                        "name": td["name"],
                        "mem_total_gb": td["total_gb"],
                        "mem_used_gb": None,
                        "util_gpu": None,
                    }
                )
            elif match.get("mem_total_gb") is None:
                match["mem_total_gb"] = td["total_gb"]

    # Classify after merging so a device with smi memory N/A is not mistaken
    # for an iGPU. GB10 is a 121 GB unified-memory device, not an iGPU.
    for row in devices:
        row["is_igpu"] = _is_igpu(row)

    env = {
        "hostname": os.uname().nodename,
        "kernel": os.uname().release,
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "cwd": str(Path.cwd()),
        "disk_free_gb": round(shutil.disk_usage(Path.cwd()).free / 1024**3, 2),
        "torch": torch_info,
        "tools": tools,
        "visible_devices": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        },
        "devices": devices,
        "cpu_offload": False,
    }

    if args.json:
        _print_json(env)
        return 0

    print(f"host: {env['hostname']}  kernel: {env['kernel']}  python: {env['python']}")
    if torch_info.get("available"):
        print(
            f"torch: {torch_info['version']}  cuda: {torch_info.get('cuda_version')}  "
            f"hip: {torch_info.get('hip_version')}  cuda_available: {torch_info.get('cuda_available')}  "
            f"devices: {torch_info.get('device_count')}"
        )
    else:
        print(f"torch: NOT INSTALLED ({torch_info.get('error')})")
    print(f"disk free: {env['disk_free_gb']} GB")
    print()
    for row in devices:
        flag = " [iGPU]" if row.get("is_igpu") else ""
        print(
            f"GPU[{row['index']}] {row.get('name')}{flag}  "
            f"mem={row.get('mem_total_gb')} GB  util={row.get('util_gpu')}%  "
            f"temp={row.get('temp_c')} C  power={row.get('power_w')} W"
        )
    if any(row.get("is_igpu") for row in devices):
        physical = [row for row in devices if not row.get("is_igpu")]
        visible = ",".join(str(row["index"]) for row in physical)
        print()
        print(f"iGPU detected; exclude it with HIP_VISIBLE_DEVICES={visible} or CUDA_VISIBLE_DEVICES={visible}")
    return 0


def _sample(device=None):
    rows = _smi(device=device)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    # Fill memory from torch when smi returns N/A (e.g. GB10 unified memory).
    try:
        import torch

        if torch.cuda.is_available():
            for row in rows:
                index = row["index"]
                if index >= torch.cuda.device_count():
                    continue
                if row.get("mem_used_gb") is None or row.get("mem_total_gb") is None:
                    free, total = torch.cuda.mem_get_info(index)
                    used = (total - free) / 1024**3
                    total_gb = total / 1024**3
                    row["mem_total_gb"] = total_gb
                    # On unified-memory devices (e.g. GB10) mem_get_info
                    # includes OS reservations and is not a useful training
                    # memory number; rely on the train log's peak_mem instead.
                    row["mem_used_gb"] = (
                        None if total_gb > 60 and used > 0.95 * total_gb else used
                    )
    except Exception:
        pass
    for row in rows:
        row["timestamp"] = timestamp
    return rows


def cmd_monitor(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    deadline = time.time() + args.duration if args.duration > 0 else None
    print(f"Monitoring GPUs every {args.interval}s" + (f" for {args.duration}s" if deadline else "") + "...")
    try:
        while deadline is None or time.time() < deadline:
            sample = _sample(device=args.device)
            if not getattr(args, "include_igpu", False):
                sample = [row for row in sample if not _is_igpu(row)]
            rows.extend(sample)
            for row in sample:
                util = row.get("util_gpu")
                mem = row.get("mem_used_gb")
                print(
                    f"{row['timestamp']} GPU[{row['index']}] "
                    f"util={util if util is not None else 'n/a'}% "
                    f"mem={f'{mem:.2f}' if mem is not None else 'n/a'} GB "
                    f"temp={row.get('temp_c', 'n/a')} C "
                    f"power={row.get('power_w', 'n/a')} W"
                )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Interrupted; writing partial capture.")

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GPU_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in GPU_FIELDS})
    print(f"Wrote {len(rows)} samples to {out}")
    return 0


def _read_monitor(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    values = sorted(values)
    p50 = statistics.median(values)
    p95 = values[min(len(values) - 1, int(0.95 * len(values)))]
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 2),
        "p50": round(p50, 2),
        "p95": round(p95, 2),
        "min": round(values[0], 2),
        "max": round(values[-1], 2),
    }


def cmd_feasibility(args):
    path = Path(args.probe_json)
    if not path.exists():
        print(f"Probe result not found: {path}")
        print(
            "Run the probe first, for example:\n"
            "  torchrun --nproc_per_node=2 train_fsdp.py --probe ...\n"
            "  gpu_doctor.py feasibility --probe-json <run>/probe_result.json"
        )
        return 1
    with open(path) as f:
        result = json.load(f)

    limit = args.limit_gb or result.get("memory_limit_gb")
    per_rank = result.get("per_rank", [])
    passed = bool(result.get("passed", False))
    if limit:
        passed = passed and all(
            float(item.get("peak_allocated_gb", 1e9)) <= limit for item in per_rank
        )
    if args.require_no_cpu_offload and result.get("cpu_offload", False):
        passed = False
    if args.min_headroom_gb > 0:
        passed = passed and all(
            float(item.get("headroom_gb", 0)) >= args.min_headroom_gb
            for item in per_rank
        )

    print(f"CPU offload: {result.get('cpu_offload', False)}")
    print(f"Optimizer:   {result.get('optimizer')}")
    print(f"World size:  {result.get('world_size')}")
    print(f"Limit:       {limit:.2f} GB/GPU" if limit else "Limit:       n/a")
    for item in per_rank:
        print(
            f"  rank {item.get('rank')}: peak={item.get('peak_allocated_gb', 0):.2f} GB "
            f"reserved={item.get('peak_reserved_gb', 0):.2f} GB "
            f"headroom={item.get('headroom_gb', 0):.2f} GB "
            f"total={item.get('total_gb', 0):.2f} GB"
        )
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}")
    if args.json:
        _print_json({"passed": passed, "probe": result})
    return 0 if passed else 1


def cmd_report(args):
    rows = _read_monitor(args.monitor_csv)
    if not getattr(args, "include_igpu", False):
        rows = [row for row in rows if not _is_igpu(row)]
    by_gpu = {}
    for row in rows:
        by_gpu.setdefault(int(row["index"]), []).append(row)

    report = {"monitor_csv": str(args.monitor_csv), "gpus": {}, "checks": {}}
    for index, gpu_rows in sorted(by_gpu.items()):
        name = next((row.get("name") for row in gpu_rows if row.get("name")), f"GPU{index}")
        util_values = [_num(row.get("util_gpu")) for row in gpu_rows]
        mem_values = [_num(row.get("mem_used_gb")) for row in gpu_rows]
        total_values = [_num(row.get("mem_total_gb")) for row in gpu_rows]
        thr_values = [_num(row.get("temp_c")) for row in gpu_rows]
        power_values = [_num(row.get("power_w")) for row in gpu_rows]
        util_values = [v for v in util_values if v is not None]
        mem_values = [v for v in mem_values if v is not None]
        total = next((v for v in total_values if v is not None), None)
        idle = sum(1 for v in util_values if v < 10) / len(util_values) if util_values else None
        report["gpus"][index] = {
            "name": name,
            "util": _stats(util_values),
            "mem_used_gb": _stats(mem_values),
            "mem_total_gb": total,
            "temp_c": _stats([v for v in thr_values if v is not None]),
            "power_w": _stats([v for v in power_values if v is not None]),
            "idle_fraction": round(idle, 3) if idle is not None else None,
        }

    util_p50 = [
        item["util"]["p50"]
        for item in report["gpus"].values()
        if item["util"] and item["util"]["p50"] is not None
    ]
    idle_fractions = [
        item["idle_fraction"]
        for item in report["gpus"].values()
        if item["idle_fraction"] is not None
    ]
    mem_ok = all(
        (item["mem_used_gb"]["max"] if item["mem_used_gb"] else 0) < 0.9 * item["mem_total_gb"]
        for item in report["gpus"].values()
        if item["mem_total_gb"]
    )
    report["checks"]["util_p50_ge_threshold"] = bool(
        util_p50 and min(util_p50) >= args.util_threshold
    )
    report["checks"]["idle_fraction_lt_threshold"] = bool(
        idle_fractions and max(idle_fractions) <= args.max_idle_fraction
    )
    report["checks"]["memory_headroom_ok"] = bool(mem_ok)
    report["checks"]["util_threshold"] = args.util_threshold
    report["checks"]["max_idle_fraction"] = args.max_idle_fraction

    if args.train_log:
        text = Path(args.train_log).read_text(errors="replace")
        throughputs = [float(x) for x in re.findall(r"throughput=([\d.]+) tok/s", text)]
        if throughputs:
            mean = statistics.mean(throughputs)
            stdev = statistics.pstdev(throughputs)
            report["throughput"] = {
                "n": len(throughputs),
                "mean_tok_s": round(mean, 1),
                "median_tok_s": round(statistics.median(throughputs), 1),
                "cv_pct": round(100 * stdev / mean, 2) if mean else None,
            }
            report["checks"]["throughput_cv_ok"] = bool(
                mean and 100 * stdev / mean < args.max_throughput_cv
            )

    if args.probe_json and Path(args.probe_json).exists():
        with open(args.probe_json) as f:
            report["probe"] = json.load(f)
        report["checks"]["probe_passed"] = bool(report["probe"].get("passed"))
        report["checks"]["cpu_offload_disabled"] = not bool(
            report["probe"].get("cpu_offload", False)
        )

    overall = all(value for key, value in report["checks"].items() if isinstance(value, bool))
    report["overall_pass"] = overall

    if args.json:
        _print_json(report)
    else:
        print(f"GPU report for {args.monitor_csv}")
        for index, item in report["gpus"].items():
            util = item["util"]
            print(
                f"GPU[{index}] {item['name']}: "
                f"util p50={util['p50'] if util else 'n/a'}% "
                f"idle={item['idle_fraction']} "
                f"mem peak={item['mem_used_gb']['max'] if item['mem_used_gb'] else 'n/a'} GB "
                f"({item['mem_total_gb'] or 'n/a'} GB) "
                f"temp max={item['temp_c']['max'] if item['temp_c'] else 'n/a'} C"
            )
        if "throughput" in report:
            print(
                f"throughput: median={report['throughput']['median_tok_s']} tok/s "
                f"CV={report['throughput']['cv_pct']}%"
            )
        print("checks:")
        for key, value in report["checks"].items():
            print(f"  {key}: {value}")
        print(f"OVERALL: {'PASS' if overall else 'FAIL'}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"Wrote {args.out}")
    return 0 if overall else 1


def _dist_worker(rank, world_size, sizes_mb, iters, port):
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    results = []
    for size_mb in sizes_mb:
        elements = max(1, int(size_mb * 1024 * 1024 // 4))
        tensor = torch.ones(elements, dtype=torch.float32, device=f"cuda:{rank}")
        for _ in range(3):
            dist.all_reduce(tensor)
        torch.cuda.synchronize(rank)
        start = time.time()
        for _ in range(iters):
            dist.all_reduce(tensor)
        torch.cuda.synchronize(rank)
        elapsed = time.time() - start
        bytes_moved = elements * 4
        # Ring all-reduce approximation: 2 * (N-1)/N * bytes.
        bus_bytes = 2 * (world_size - 1) / world_size * bytes_moved * iters
        bandwidth = bus_bytes / elapsed / 1024**3 if elapsed > 0 else 0
        results.append({"size_mb": size_mb, "bandwidth_gbs": bandwidth})
    dist.barrier()
    dist.destroy_process_group()
    return results


def _dist_entry(rank, world_size, sizes_mb, iters, port, queue):
    """Top-level entry point so the spawn context can pickle it."""
    queue.put((rank, _dist_worker(rank, world_size, sizes_mb, iters, port)))


def cmd_dist(args):
    try:
        import torch
        import torch.multiprocessing as mp
    except Exception as exc:  # noqa: BLE001
        print(f"PyTorch unavailable: {exc}")
        return 1
    if not torch.cuda.is_available():
        print("CUDA/ROCm is not available.")
        return 1
    world_size = args.nproc or torch.cuda.device_count()
    if world_size < 1:
        print("No GPU available.")
        return 1
    port = args.port
    sizes = [float(x) for x in args.sizes]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    mp.spawn(
        _dist_entry,
        args=(world_size, sizes, args.iters, port, queue),
        nprocs=world_size,
        join=True,
    )
    results = {}
    while not queue.empty():
        rank, data = queue.get()
        results[rank] = data

    print(f"All-reduce benchmark: world_size={world_size}, iters={args.iters}")
    for size in sizes:
        values = [
            next(item["bandwidth_gbs"] for item in data if item["size_mb"] == size)
            for data in results.values()
        ]
        if values:
            print(f"  {size:g} MB: mean={statistics.mean(values):.1f} GB/s")
    if args.json:
        _print_json(results)
    return 0


def cmd_sweep(args):
    """Run short single-GPU configurations and parse throughput from train.log."""
    script = Path(args.script)
    if not script.exists():
        print(f"Training script not found: {script}")
        return 1
    base = shlex.split(args.base_args)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for batch in args.batch_sizes:
        for grad_accum in args.grad_accums:
            for workers in args.num_workers:
                run_dir = out_root / f"b{batch}_ga{grad_accum}_w{workers}"
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                cmd = [
                    sys.executable,
                    str(script),
                    *base,
                    "--batch_size", str(batch),
                    "--gradient_accumulation_steps", str(grad_accum),
                    "--num_workers", str(workers),
                    "--max_samples", str(args.max_samples),
                    "--output_dir", str(run_dir),
                    "--save_steps", "0",
                    "--eval_steps", "1000000",
                    "--log_steps", "1",
                    "--num_qualitative_samples", "0",
                    "--no_checkpoints",
                ]
                print(f"=== batch={batch} ga={grad_accum} workers={workers} ===")
                proc = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, timeout=args.timeout,
                )
                log = proc.stdout or ""
                throughputs = [float(x) for x in re.findall(r"throughput=([\d.]+) tok/s", log)]
                oom = "out of memory" in log.lower() or "OutOfMemoryError" in log
                result = {
                    "batch_size": batch,
                    "grad_accum": grad_accum,
                    "num_workers": workers,
                    "returncode": proc.returncode,
                    "oom": oom,
                    "median_tok_s": round(statistics.median(throughputs), 1) if throughputs else None,
                    "n_steps": len(throughputs),
                }
                results.append(result)
                print(json.dumps(result))
    csv_path = out_root / "sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {csv_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", help="Inventory drivers, PyTorch and GPUs")
    env.add_argument("--json", action="store_true")
    env.set_defaults(func=cmd_env)

    monitor = sub.add_parser("monitor", help="Sample GPU metrics to CSV")
    monitor.add_argument("--interval", type=float, default=1.0)
    monitor.add_argument("--duration", type=float, default=0.0,
                         help="Seconds to monitor; 0 means until Ctrl-C")
    monitor.add_argument("--device", type=int, default=None)
    monitor.add_argument("--include-igpu", action="store_true",
                         help="Also sample integrated GPUs (excluded by default).")
    monitor.add_argument("--out", default="outputs/gpu_monitor.csv")
    monitor.set_defaults(func=cmd_monitor)

    feasibility = sub.add_parser("feasibility", help="Read a train_fsdp.py probe result")
    feasibility.add_argument("--probe-json", required=True)
    feasibility.add_argument("--limit-gb", type=float, default=0.0)
    feasibility.add_argument("--min-headroom-gb", type=float, default=0.5)
    feasibility.add_argument("--require-no-cpu-offload", action="store_true", default=True)
    feasibility.add_argument("--json", action="store_true")
    feasibility.set_defaults(func=cmd_feasibility)

    report = sub.add_parser("report", help="Summarize a monitor CSV")
    report.add_argument("--monitor-csv", required=True)
    report.add_argument("--train-log", default=None)
    report.add_argument("--probe-json", default=None)
    report.add_argument("--util-threshold", type=float, default=85.0)
    report.add_argument("--max-idle-fraction", type=float, default=0.10)
    report.add_argument("--max-throughput-cv", type=float, default=5.0)
    report.add_argument("--include-igpu", action="store_true",
                        help="Also report integrated GPUs (excluded by default).")
    report.add_argument("--out", default=None)
    report.add_argument("--json", action="store_true")
    report.set_defaults(func=cmd_report)

    dist = sub.add_parser("dist", help="Measure all-reduce bandwidth")
    dist.add_argument("--nproc", type=int, default=None)
    dist.add_argument("--sizes", nargs="+", default=["1", "16", "64", "256"])
    dist.add_argument("--iters", type=int, default=20)
    dist.add_argument("--port", type=int, default=29511)
    dist.add_argument("--json", action="store_true")
    dist.set_defaults(func=cmd_dist)

    sweep = sub.add_parser("sweep", help="Short single-GPU configuration sweep")
    sweep.add_argument("--script", default="train.py")
    sweep.add_argument("--base-args", required=True,
                       help="Quoted training arguments (without batch/accum/workers)")
    sweep.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    sweep.add_argument("--grad-accums", nargs="+", type=int, default=[48, 24, 12, 6])
    sweep.add_argument("--num-workers", nargs="+", type=int, default=[0, 4, 8])
    sweep.add_argument("--max-samples", type=int, default=96)
    sweep.add_argument("--output-root", default="outputs/gpu_sweep")
    sweep.add_argument("--timeout", type=int, default=900)
    sweep.set_defaults(func=cmd_sweep)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
