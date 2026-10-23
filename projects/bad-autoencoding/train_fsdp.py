#!/usr/bin/env python3
"""GPU-only FSDP training for the vision-compression regime.

This entry point is for hardware where the 3B decoder does not fit on a single
GPU. It shards parameters, gradients and optimizer state across ranks with
FullyShardedDataParallel (FSDP). There is intentionally no CPU offload: all
model, gradient, optimizer and activation memory stays on the GPUs.

The existing single-GPU ``train.py`` remains the reference implementation.
This script currently supports the ``vision`` regime only.

Example (2 GPUs, effective batch 48):
    torchrun --nproc_per_node=2 train_fsdp.py \
        --regime vision --vision_mode tiny --vision_prompt free_ocr \
        --no-train-encoder --objective lm \
        --data_path data/training/full/train_arrow \
        --val_data_path data/training/full/val_arrow \
        --output_dir outputs/vision_tiny_frozen_fsdp \
        --batch_size 1 --gradient_accumulation_steps 24 \
        --optimizer adamw8bit --use_decoder_checkpointing \
        --attn_implementation eager --eval_steps 500 --save_steps 0 \
        --num_qualitative_samples 0 --save_val_per_sample

Memory probe (one forward/backward/optimizer step, then exit):
    torchrun --nproc_per_node=2 train_fsdp.py --probe ...same flags...
"""

import argparse
import functools
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler

from training.arguments import create_argument_parser
from trainers import VisionCompressionTrainer, load_model_and_tokenizer
from trainers.config import VISION_PROMPT_PRESETS

# Imported lazily to avoid circular-import surprises at module import time.
# train.py defines the dataset/collate/evaluate helpers used by both entry points.


def _build_parser() -> argparse.ArgumentParser:
    parser = create_argument_parser()
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adamw8bit", "adafactor"],
        default="adamw8bit",
        help="Optimizer used for the all-GPU FSDP run. adamw8bit is the "
             "memory-efficient default; adafactor is a last-resort fallback "
             "that is not faithful to the paper's AdamW recipe.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Run one forward/backward/optimizer step, write probe_result.json, "
             "and exit without training.",
    )
    parser.add_argument(
        "--probe_memory_limit_gb",
        type=float,
        default=0.0,
        help="Fail the probe if peak allocated memory exceeds this per-GPU "
             "limit. 0 uses 90%% of the device's total memory.",
    )
    parser.add_argument(
        "--target_effective_batch",
        type=int,
        default=48,
        help="Expected batch_size * gradient_accumulation_steps * world_size. "
             "The script refuses to silently run a different effective batch.",
    )
    return parser


def _setup_logger(output_dir: Path, rank: int) -> logging.Logger:
    logger = logging.getLogger(f"train_fsdp_rank{rank}")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / "train.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def _resolve_vision_prompt(value):
    if value and value in VISION_PROMPT_PRESETS:
        return VISION_PROMPT_PRESETS[value]
    return value


def _build_optimizer(model: torch.nn.Module, args, logger: logging.Logger):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found after freezing.")

    if args.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError(
                "adamw8bit requested but bitsandbytes is unavailable. "
                "Use --optimizer adamw or --optimizer adafactor explicitly."
            ) from exc
        logger.info(f"Using AdamW8bit over {sum(p.numel() for p in params):,} parameters")
        return bnb.optim.AdamW8bit(
            params, lr=args.learning_rate, weight_decay=args.weight_decay
        )

    if args.optimizer == "adafactor":
        from transformers.optimization import Adafactor

        logger.warning(
            "Using Adafactor. This is a memory-saving fallback, NOT the "
            "paper's AdamW recipe; results must be labelled accordingly."
        )
        return Adafactor(
            params,
            lr=args.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=args.weight_decay,
        )

    logger.info(f"Using AdamW over {sum(p.numel() for p in params):,} parameters")
    try:
        return torch.optim.AdamW(
            params, lr=args.learning_rate, weight_decay=args.weight_decay, fused=True
        )
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(
            params, lr=args.learning_rate, weight_decay=args.weight_decay
        )


def _build_scheduler(optimizer, num_training_steps: int, warmup_ratio: float, logger):
    warmup_steps = int(num_training_steps * warmup_ratio)
    if warmup_steps >= num_training_steps:
        logger.warning(
            f"Warmup steps ({warmup_steps}) >= total steps ({num_training_steps}); "
            "disabling warmup."
        )
        warmup_steps = 0

    if warmup_steps > 0:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                         total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=num_training_steps - warmup_steps),
            ],
            milestones=[warmup_steps],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
    logger.info(
        f"Scheduler: warmup={warmup_steps}, total_steps={num_training_steps}"
    )
    return scheduler


def _save_checkpoint(
    model,
    optimizer,
    args,
    path: Path,
    val_loss,
    val_perplexity,
    global_step: int,
    logger: logging.Logger,
) -> None:
    """Gather a full state dict on all ranks; only rank 0 writes the file.

    The full-state-dict gather is a collective, so every rank must call this.
    Optimizer state is intentionally omitted: the FSDP entry point has no
    mid-epoch resume, and the single-GPU train.py only needs model weights for
    --init_from_checkpoint / --validation_only.
    """
    full_state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_config):
        model_state = model.state_dict()

    if dist.get_rank() != 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    # Defensive: FULL_STATE_DICT normally strips wrapper prefixes, but keep the
    # checkpoint loadable by the single-GPU train.py path.
    model_state = {
        key.replace("_fsdp_wrapped_module.", ""): value
        for key, value in model_state.items()
    }

    checkpoint = {
        "format_version": 2.0,
        "epoch": args.num_epochs - 1,
        "batch_idx": -1,
        "global_step": global_step,
        "model_state_dict": model_state,
        "optimizer_state_dict": None,
        "scheduler_state_dict": None,
        "args": vars(args),
        "regime": args.regime,
        "objective": args.objective,
        "vision_mode": args.vision_mode,
        "train_encoder": args.train_encoder,
        "optimizer": args.optimizer,
        "fsdp": True,
        "world_size": dist.get_world_size(),
        "cpu_offload": False,
        "val_loss": val_loss,
        "val_perplexity": val_perplexity,
        "best_val_loss": val_loss,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    torch.save(checkpoint, path)
    logger.info(f"Saved FSDP checkpoint to {path}")


def _probe(
    trainer,
    fsdp_model,
    optimizer,
    train_loader,
    args,
    logger: logging.Logger,
    output_dir: Path,
) -> int:
    local_rank = torch.cuda.current_device()
    rank = dist.get_rank()

    fsdp_model.train()
    batch = next(iter(train_loader))
    image = batch["image"]
    target = batch["continuation"] if args.objective == "lm" else batch["context"]
    hybrid_text = batch.get("hybrid_text", None)

    torch.cuda.reset_peak_memory_stats(local_rank)
    loss, _ = trainer.forward(image, target, hybrid_text, objective=args.objective)
    loss.backward()
    FSDP.clip_grad_norm_(fsdp_model, args.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(local_rank)

    peak_alloc = torch.cuda.max_memory_allocated(local_rank)
    peak_reserved = torch.cuda.max_memory_reserved(local_rank)
    free_bytes, total_bytes = torch.cuda.mem_get_info(local_rank)
    local = {
        "rank": rank,
        "device": torch.cuda.get_device_name(local_rank),
        "peak_allocated_gb": peak_alloc / 1024**3,
        "peak_reserved_gb": peak_reserved / 1024**3,
        "free_gb": free_bytes / 1024**3,
        "total_gb": total_bytes / 1024**3,
        "headroom_gb": (total_bytes - peak_alloc) / 1024**3,
        "cpu_offload": False,
    }

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)

    if rank == 0:
        limit_gb = args.probe_memory_limit_gb
        if limit_gb <= 0:
            limit_gb = 0.9 * min(item["total_gb"] for item in gathered)
        passed = all(item["peak_allocated_gb"] <= limit_gb for item in gathered)
        result = {
            "passed": passed,
            "memory_limit_gb": limit_gb,
            "world_size": dist.get_world_size(),
            "cpu_offload": False,
            "optimizer": args.optimizer,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "per_rank": gathered,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "probe_result.json", "w") as f:
            json.dump(result, f, indent=2)
        for item in gathered:
            logger.info(
                "rank %s: peak_alloc=%.2f GB peak_reserved=%.2f GB "
                "headroom=%.2f GB (%.1f/%.1f GB) cpu_offload=false",
                item["rank"],
                item["peak_allocated_gb"],
                item["peak_reserved_gb"],
                item["headroom_gb"],
                item["peak_allocated_gb"],
                item["total_gb"],
            )
        logger.info(
            "PROBE %s: peak limit %.2f GB/GPU",
            "PASS" if passed else "FAIL",
            limit_gb,
        )
        return 0 if passed else 1
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.regime != "vision":
        raise SystemExit(
            "train_fsdp.py currently supports --regime vision only. "
            "Use train.py for text/meanpool/conv1d_residual."
        )
    if args.validation_only:
        raise SystemExit(
            "FSDP validation-only is not supported; use train.py --validation_only."
        )
    if args.resume_from_checkpoint or args.resume or args.init_from_checkpoint:
        raise SystemExit(
            "The FSDP entry point does not implement checkpoint resume/init yet."
        )
    if args.use_optimized_model:
        raise SystemExit(
            "FSDP currently supports the original model implementation; "
            "use --no-use-optimized-model."
        )
    if args.compile:
        raise SystemExit("torch.compile is not supported by the FSDP entry point.")

    if "LOCAL_RANK" not in os.environ:
        raise SystemExit(
            "train_fsdp.py must be launched with torchrun, e.g. "
            "torchrun --nproc_per_node=2 train_fsdp.py ..."
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm is not available to PyTorch.")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 1:
        raise SystemExit("WORLD_SIZE must be >= 1.")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    if args.output_dir is None:
        raise SystemExit("--output_dir is required.")
    output_dir = Path(args.output_dir)
    logger = _setup_logger(output_dir, rank)

    # Keep the run aligned with the paper's effective batch.
    effective_batch = args.batch_size * args.gradient_accumulation_steps * world_size
    if effective_batch != args.target_effective_batch:
        raise SystemExit(
            f"Effective batch is {effective_batch} "
            f"({args.batch_size} x {args.gradient_accumulation_steps} x {world_size}), "
            f"expected {args.target_effective_batch}. Adjust the flags explicitly."
        )

    seed = args.seed if args.seed is not None else 42
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    args.use_wandb = False
    args.vision_prompt = _resolve_vision_prompt(args.vision_prompt)
    if args.device is None:
        args.device = f"cuda:{local_rank}"

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        logger.info(
            "FSDP run: world_size=%s local_rank=%s device=%s cpu_offload=false",
            world_size,
            local_rank,
            torch.cuda.get_device_name(local_rank),
        )
        logger.info(
            "Effective batch=%s (%s x %s x %s), optimizer=%s",
            effective_batch,
            args.batch_size,
            args.gradient_accumulation_steps,
            world_size,
            args.optimizer,
        )

    # Imported here so load_model_and_tokenizer can register the vendored package.
    from train import create_vision_dataset, evaluate, vision_collate_fn

    model, tokenizer = load_model_and_tokenizer(
        device=f"cuda:{local_rank}",
        use_optimized_model=False,
        use_encoder_checkpointing=False,
        load_vision_encoder=True,
        attn_implementation=args.attn_implementation,
    )

    if args.use_decoder_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if rank == 0:
            logger.info("Decoder gradient checkpointing enabled")

    trainer = VisionCompressionTrainer(
        model,
        tokenizer,
        vision_mode=args.vision_mode,
        device=f"cuda:{local_rank}",
        hybrid_text_tokens=args.hybrid_text_tokens,
        vision_prompt=args.vision_prompt,
        train_encoder=args.train_encoder,
    )

    try:
        from deepseek_ocr.modeling_deepseekv2 import DeepseekV2DecoderLayer
    except ImportError as exc:
        raise SystemExit(
            "Could not import the vendored decoder layer class for FSDP wrapping."
        ) from exc

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={DeepseekV2DecoderLayer},
    )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    fsdp_model = FSDP(
        trainer.model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        sync_module_states=True,
        # Overlap the next unit's all-gather with the current unit's compute.
        # This matters for the 1-sample micro-batches used on 16 GB GPUs.
        forward_prefetch=True,
    )
    trainer.model = fsdp_model

    if rank == 0:
        logger.info("FSDP FULL_SHARD wrapping complete (no CPU offload)")

    optimizer = _build_optimizer(fsdp_model, args, logger)

    train_dataset = create_vision_dataset(
        args.data_path,
        tokenizer,
        args.vision_mode,
        max_samples=args.max_samples,
        hybrid_text_tokens=args.hybrid_text_tokens,
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "sampler": train_sampler,
        "collate_fn": vision_collate_fn,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_dataset, **loader_kwargs)

    if rank == 0:
        logger.info(
            "Train dataset: %s samples, %s batches per rank",
            len(train_dataset),
            len(train_loader),
        )

    if args.probe:
        if len(train_loader) == 0:
            raise SystemExit("Probe needs at least one batch per rank.")
        status = _probe(
            trainer, fsdp_model, optimizer, train_loader, args, logger, output_dir
        )
        dist.barrier()
        dist.destroy_process_group()
        return status

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    num_training_steps = steps_per_epoch * args.num_epochs
    scheduler = _build_scheduler(
        optimizer, num_training_steps, args.warmup_ratio, logger
    )

    val_loader = None
    if args.val_data_path:
        # FSDP forward passes are collective, so every rank must run the same
        # validation batches. Each rank evaluates the full validation split;
        # only rank 0 logs and writes per-sample results.
        val_dataset = create_vision_dataset(
            args.val_data_path,
            tokenizer,
            args.vision_mode,
            max_samples=None,
            hybrid_text_tokens=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=vision_collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if rank == 0:
            logger.info("Validation dataset: %s samples", len(val_dataset))

    best_val_loss = float("inf")
    global_step = 0
    best_path = output_dir / "best_checkpoint.pt"
    final_path = output_dir / "final_checkpoint.pt"

    if args.initial_validation and val_loader is not None:
        if rank == 0:
            logger.info("Running initial validation (all ranks)...")
        metrics = evaluate(
            trainer, val_loader, "vision", logger, args, global_step=0,
            num_qualitative_samples=0,
            max_generation_tokens=args.max_generation_tokens,
            output_dir=output_dir if rank == 0 else None,
            eval_seed=args.eval_seed,
            collect_per_sample=False,
        )
        if rank == 0:
            best_val_loss = metrics["loss"]
            logger.info(
                "Initial validation: loss=%.4f, ppl=%.2f",
                metrics["loss"],
                metrics["perplexity"],
            )
        torch.cuda.empty_cache()
        dist.barrier()

    logger.info("Starting FSDP training loop...")
    epoch_start = time.time()
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        fsdp_model.train()
        accumulated_loss = 0.0
        accumulated_batches = 0
        tokens_since_log = 0
        log_start = time.time()
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            image = batch["image"]
            target = (
                batch["continuation"]
                if args.objective == "lm"
                else batch["context"]
            )
            hybrid_text = batch.get("hybrid_text", None)

            loss, labels = trainer.forward(
                image, target, hybrid_text, objective=args.objective
            )
            target_tokens = int((labels[0] != -100).sum().item())
            tokens_since_log += target_tokens * labels.shape[0] * world_size

            accumulated_loss += loss.detach()
            accumulated_batches += 1
            num_batches += 1

            (loss / args.gradient_accumulation_steps).backward()

            if accumulated_batches % args.gradient_accumulation_steps == 0:
                grad_norm = FSDP.clip_grad_norm_(
                    fsdp_model, args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if global_step % args.log_steps == 0:
                    elapsed = time.time() - log_start
                    avg_loss = accumulated_loss.item() / accumulated_batches
                    throughput = tokens_since_log / max(elapsed, 1e-9)
                    logger.info(
                        "Epoch %s Step %s (Global: %s): loss=%.4f, ppl=%.2f, "
                        "grad_norm=%.2f, lr=%.2e, throughput=%.0f tok/s, "
                        "peak_mem=%.2f GB",
                        epoch + 1,
                        global_step,
                        global_step,
                        avg_loss,
                        math.exp(avg_loss) if avg_loss < 50 else float("inf"),
                        float(grad_norm),
                        optimizer.param_groups[0]["lr"],
                        throughput,
                        torch.cuda.max_memory_allocated(local_rank) / 1024**3,
                    )
                    accumulated_loss = 0.0
                    accumulated_batches = 0
                    tokens_since_log = 0
                    log_start = time.time()

                if (
                    args.eval_steps > 0
                    and global_step % args.eval_steps == 0
                    and val_loader is not None
                ):
                    metrics = evaluate(
                        trainer, val_loader, "vision", logger, args,
                        global_step=global_step,
                        num_qualitative_samples=0,
                        max_generation_tokens=args.max_generation_tokens,
                        output_dir=output_dir if rank == 0 else None,
                        eval_seed=args.eval_seed,
                        collect_per_sample=False,
                    )
                    if rank == 0:
                        logger.info(
                            "Validation at step %s: loss=%.4f, ppl=%.2f",
                            global_step,
                            metrics["loss"],
                            metrics["perplexity"],
                        )
                        if metrics["loss"] < best_val_loss:
                            best_val_loss = metrics["loss"]
                        torch.cuda.empty_cache()
                    dist.barrier()

                # Non-resumable periodic checkpoint: limits the loss if the
                # multi-day run crashes. The full-state-dict gather is
                # collective, so every rank must enter it.
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    _save_checkpoint(
                        fsdp_model,
                        optimizer,
                        args,
                        output_dir / f"checkpoint_step_{global_step}.pt",
                        None,
                        None,
                        global_step,
                        logger,
                    )

        # Flush the remainder of a partial accumulation window.
        if accumulated_batches % args.gradient_accumulation_steps != 0:
            FSDP.clip_grad_norm_(fsdp_model, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

    epoch_seconds = time.time() - epoch_start
    logger.info(
        "Training complete: %s steps in %.1f s (%.1f s/step)",
        global_step,
        epoch_seconds,
        epoch_seconds / max(global_step, 1),
    )

    final_loss = None
    final_ppl = None
    if val_loader is not None:
        if rank == 0:
            logger.info("Running final validation (all ranks)...")
        metrics = evaluate(
            trainer, val_loader, "vision", logger, args,
            global_step=global_step,
            num_qualitative_samples=0,
            max_generation_tokens=args.max_generation_tokens,
            output_dir=output_dir if rank == 0 else None,
            eval_seed=args.eval_seed,
            collect_per_sample=args.save_val_per_sample and rank == 0,
        )
        if rank == 0:
            final_loss = metrics["loss"]
            final_ppl = metrics["perplexity"]
            logger.info(
                "Final validation: loss=%.4f, ppl=%.2f",
                final_loss,
                final_ppl,
            )
            if final_loss < best_val_loss:
                best_val_loss = final_loss

    # The full-state-dict gather is collective: every rank must enter it.
    payload = [final_loss, final_ppl]
    dist.broadcast_object_list(payload, src=0)
    final_loss, final_ppl = payload

    if not args.no_checkpoints:
        _save_checkpoint(
            fsdp_model, optimizer, args, final_path,
            final_loss, final_ppl, global_step, logger,
        )
        if rank == 0:
            # Provide best_checkpoint.pt for report/extractor tooling.
            if best_path.exists() or best_path.is_symlink():
                best_path.unlink()
            try:
                os.symlink(final_path.name, best_path)
            except OSError:
                pass
    dist.barrier()

    if rank == 0:
        summary = {
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "best_checkpoint": str(best_path),
            "final_checkpoint": str(final_path),
            "optimizer": args.optimizer,
            "cpu_offload": False,
            "world_size": world_size,
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Summary written to %s", output_dir / "summary.json")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
