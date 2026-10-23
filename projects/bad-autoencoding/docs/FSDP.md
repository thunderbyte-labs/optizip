# Multi-GPU FSDP training

`train_fsdp.py` is a GPU-only FullyShardedDataParallel (FSDP) entry point for
the vision regime. Use it when the 3B decoder does not fit on a single GPU.
The reference single-GPU path remains `train.py`.

## Why FSDP and not DDP

- **DDP** replicates parameters, gradients and optimizer state on every GPU. It
  does not reduce per-GPU memory.
- **FSDP** shards those tensors across ranks and all-gathers them layer by
  layer. It is the right tool for a model that fits on the combined memory but
  not on one device.

There is intentionally **no CPU offload** in `train_fsdp.py`: all model,
gradient, optimizer and activation memory stays on the GPUs. If the all-GPU
probe fails, use validation-only (`train.py --validation_only`) or more GPUs.

## Effective batch

```
effective_batch = batch_size × gradient_accumulation_steps × world_size
```

The script refuses to start if this does not equal `--target_effective_batch`
(default 48). For two 16 GB GPUs:

```text
1 × 24 × 2 = 48
```

## Memory guidance

| Per-GPU memory | Suggested configuration |
|---:|---|
| 80 GB | `batch_size 8 × grad_accum 6 × 1` (single GPU, `train.py`) |
| 48 GB | `batch_size 8 × grad_accum 6 × 1` |
| 24 GB | `train.py` with `batch_size 1 × grad_accum 48`, decoder checkpointing, 8-bit optimizer |
| 16 GB | `train_fsdp.py` with `batch_size 1 × grad_accum 24 × 2`, decoder checkpointing, 8-bit optimizer |

The 16 GB configuration is tight. Always run the probe first; it reports peak
allocated memory and headroom per rank.

## Probe

```bash
torchrun --nproc_per_node=2 train_fsdp.py --probe \
    --regime vision --vision_mode tiny --vision_prompt free_ocr \
    --no-train-encoder --objective lm \
    --data_path data/training/full/train_arrow \
    --val_data_path data/training/full/val_arrow \
    --output_dir outputs/fsdp_probe \
    --max_samples 8 \
    --batch_size 1 --gradient_accumulation_steps 24 \
    --target_effective_batch 48 \
    --optimizer adamw8bit --use_decoder_checkpointing \
    --attn_implementation eager --num_workers 0 --no_checkpoints
```

The probe writes `probe_result.json`:

```json
{
  "passed": true,
  "cpu_offload": false,
  "per_rank": [{"rank": 0, "peak_allocated_gb": 13.2, "headroom_gb": 2.7}]
}
```

## Full run

```bash
torchrun --nproc_per_node=2 train_fsdp.py \
    --regime vision --vision_mode tiny --vision_prompt free_ocr \
    --no-train-encoder --objective lm \
    --data_path data/training/full/train_arrow \
    --val_data_path data/training/full/val_arrow \
    --output_dir outputs/vision_tiny_frozen_fsdp \
    --batch_size 1 --gradient_accumulation_steps 24 \
    --target_effective_batch 48 \
    --optimizer adamw8bit --use_decoder_checkpointing \
    --attn_implementation eager \
    --num_epochs 1 --learning_rate 1e-4 \
    --eval_steps 500 --save_steps 0 \
    --num_qualitative_samples 0 --save_val_per_sample
```

The launcher `scripts/reproduce_plan1/04_vision_tiny_frozen_fsdp.sh` wraps this
command with environment variables for data/output paths.

## Optimizers

- `adamw8bit` (default): bitsandbytes AdamW8bit, required to fit on 16 GB.
- `adamw`: standard fused AdamW; only for larger GPUs.
- `adafactor`: a memory fallback. **It is not the paper's AdamW recipe**; any
  run using it must be labelled as non-faithful to the paper.

If `adamw8bit` is unavailable on a ROCm build, do not silently switch to
Adafactor: the launcher warns and the results must be reported separately.

## Checkpoints

- Only rank 0 writes checkpoints.
- `best_checkpoint.pt` and `final_checkpoint.pt` contain a full state dict
  (FSDP `FULL_STATE_DICT`), loadable by the single-GPU `train.py`
  `--init_from_checkpoint` path.
- Optimizer state is gathered on rank 0; no CPU offload is used.
- Mid-epoch resume is not implemented in `train_fsdp.py`.

## Validation

- Validation runs on rank 0; other ranks wait at a barrier.
- `--save_val_per_sample` writes `val_per_sample.jsonl`, usable by
  `scripts/reproduce_plan1/estimate_paper_metric.py`.
- Qualitative generation is disabled in the FSDP path.

## ROCm

- PyTorch ROCm uses the `nccl` backend name (RCCL underneath).
- Use `HIP_VISIBLE_DEVICES` to select the discrete GPUs and exclude integrated
  GPUs.
- FlashAttention-2 may be unavailable; `--attn_implementation eager` is the
  supported fallback.

## Monitoring

`scripts/reproduce_plan1/gpu_doctor.py` provides:

```bash
gpu_doctor.py env
gpu_doctor.py monitor --interval 1 --duration 600 --out outputs/gpu_monitor.csv
gpu_doctor.py feasibility --probe-json outputs/fsdp_probe/probe_result.json
gpu_doctor.py report --monitor-csv outputs/gpu_monitor.csv \
    --train-log outputs/vision_tiny_frozen_fsdp/train.log \
    --probe-json outputs/fsdp_probe/probe_result.json
```

The report checks GPU utilization, idle fraction, memory headroom, throughput
variation and the probe result.
