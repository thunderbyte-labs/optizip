# PLAN_1 — Language-modeling verification

Scripts to reproduce two Table 2 rows of the paper:

| Run | Regime | Key settings | Paper tokens | Paper PPL |
|---|---|---|---|---|
| Hierarchical t=63 | `conv1d_residual` | `compression_target=63`, recon init | 65 | **5.14** |
| Vision tiny frozen | `vision` | `vision_mode=tiny`, `free_ocr` prompt, frozen encoder | 78 | **5.35** |

Both runs use `objective=lm`, `hybrid_text_tokens=0`, one epoch over 500k
samples and an effective batch size of 48.

## Quick start

```bash
cd /path/to/bad-autoencoding

# 1. Environment, weights, reconstruction checkpoint, Arrow datasets
bash scripts/reproduce_plan1/00_setup_gb10.sh

# 2. Fast smoke tests (model loading, image pipeline, context lengths)
bash scripts/reproduce_plan1/01_smoke_vision.sh
bash scripts/reproduce_plan1/01_smoke_conv1d.sh

# 3. Full runs (sequential; each one takes about four days on a GB10)
bash scripts/reproduce_plan1/02_conv1d_t63.sh
bash scripts/reproduce_plan1/03_vision_tiny_frozen.sh

# 4. Formatted results
uv run python scripts/reproduce_plan1/extract_results.py

# 5. Statistical report (Markdown + figures)
uv run python scripts/reproduce_plan1/make_report.py
```

The report is written to `outputs/report/REPORT.md`, with PNG figures and a
machine-readable `summary.json`.

## Why these flags matter

### Hierarchical t=63

The Table 2 row is `Enc Init = Recon.`, `Dec Init = Recon.`. Starting from the
pretrained DeepSeek-OCR decoder is **not** the same experiment. The script
loads the released reconstruction checkpoint:

```text
ivnle/bad-autoencoding/conv1d_t63_h0_recon/best_checkpoint.pt
```

and then fine-tunes with:

```bash
--init_from_checkpoint <recon checkpoint> \
--allow_objective_switch \
--no-train-encoder
```

This mirrors the authors' released t=250 LM run, which used the same two-stage
recipe and reached PPL 4.96 (Table 2).

### Vision tiny frozen

The paper's row is `Enc Init = DS (frozen)`, `Dec Init = DS`. Two details are
easy to miss:

```bash
--no-train-encoder        # otherwise the vision encoder is fine-tuned
--vision_prompt free_ocr  # "\nFree OCR." is exactly 4 tokens
```

Together with BOS, this gives `73 + 4 + 1 = 78` context tokens, hence the
paper's 12.8x compression. Without the prompt the run has 74 context tokens and
is not comparable to the 5.35 target.

## Expected log lines

Conv1d t=63:

```text
Effective context tokens (per-sample): 65
Target tokens per sample: 1000
Compression ratio: 15.38x
```

Vision tiny frozen:

```text
Effective context tokens (per-sample): 78
Target tokens per sample: 1000
Compression ratio: 12.82x
```

The smoke scripts fail with a clear error if these values are missing.

## Environment variables

All scripts source `common.sh`. Override any of these before running:

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIR` | `./data/training/full` | JSONL + `images/` root |
| `TRAIN_DATA` | `$DATA_DIR/train_arrow` | Arrow training split |
| `VAL_DATA` | `$DATA_DIR/val_arrow` | Arrow validation split |
| `CKPT_DIR` | `./checkpoints/bad-autoencoding` | Downloaded checkpoints |
| `OUTPUT_ROOT` | `./outputs` | Run output root |
| `BATCH_SIZE` | `8` | Per-device batch size |
| `GRAD_ACCUM` | `6` | Gradient accumulation (8 x 6 = 48) |
| `EVAL_STEPS` | `500` | Validation interval |
| `NUM_WORKERS` | `8` | DataLoader workers |
| `ATTN_IMPLEMENTATION` | `eager` | `auto`, `flash_attention_2`, or `eager` |
| `FORCE` | `0` | Set to `1` to delete an existing run directory |

Example with a larger batch:

```bash
BATCH_SIZE=12 GRAD_ACCUM=4 bash scripts/reproduce_plan1/02_conv1d_t63.sh
```

## Outputs

Each run writes:

```text
outputs/<run>/train.log              # full log, including final validation
outputs/<run>/args.json              # exact configuration
outputs/<run>/val_per_sample.jsonl   # per-sample NLL from the final validation
outputs/<run>/best_checkpoint.pt     # best validation loss
outputs/<run>/final_checkpoint.pt    # symlink to best when final is best
```

`extract_results.py` prints the `=== LM RUN ===`, `=== VAL FINAL ===` and
`=== VS PAPER ===` blocks, including:

- `tokens_seen_by_decoder = 500000 x (context_tokens + 1000)`
  - conv1d t=63: **532,500,000**
  - vision tiny: **539,000,000**
- compression ratio `1000 / context_tokens`
- `delta_ppl` against the paper value.

## Statistical report

The full run scripts pass `--save_val_per_sample`, which writes the mean NLL of
each validation sample to `val_per_sample.jsonl` during the final validation.
`make_report.py` uses those values to produce:

- `REPORT.md`: results table, confidence intervals, paired comparison, and a
  discussion of what the differences do and do not establish.
- `summary.json`: machine-readable metrics.
- `fig1_ppl_vs_paper.png`: our PPL with 95% bootstrap CI next to the paper.
- `fig2_delta_vs_paper.png`: signed difference from the paper.
- `fig3_bootstrap_ppl.png`: bootstrap distribution of our PPL.
- `fig4_paired_delta.png`: paired ΔPPL/ΔNLL on the same validation samples.
- `fig5_validation_curves.png`: PPL during training.
- `fig6_per_sample_ecdf.png`: per-sample loss distributions.

The within-run CI reflects validation-set sampling noise, not training-seed
variance. The paper publishes point estimates only, so the paper comparison is
descriptive; the hierarchical-vs-vision comparison is a proper paired test on
the shared validation samples.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Torch not compiled with CUDA enabled` | `uv sync` did not install the CUDA wheel; check `torch==2.13.0` in `pyproject.toml`. |
| `FlashAttention2 has been toggled on, but ...` | Run with `--attn_implementation eager` (or update to this branch). |
| `Effective context tokens: 74` for vision | `--vision_prompt free_ocr` is missing. |
| PPL far above 5.14 for conv1d | `--init_from_checkpoint` is missing or points to the wrong checkpoint. |
| CUDA OOM | Lower `BATCH_SIZE` and raise `GRAD_ACCUM` to keep 48. |
| Checkpoint download takes long | The reconstruction checkpoint is 18.7 GB; it is reused on subsequent runs. |

## Expected duration on a GB10

With eager attention and `batch_size=8`, the vision tiny smoke benchmark
measured about **1.7k target tokens/s** (roughly 4.8 s per batch of 8 samples,
or ~30 s per optimizer update at effective batch 48). One 10,417-update epoch
therefore takes about **3.5 days**, plus validation passes. Plan for roughly
four days per run, and run the two configurations sequentially.

`batch_size=16` did not improve throughput in the same benchmark, so keep the
default 8 x 6 split unless a different GPU changes the picture.

## Notes

- `--save_steps 0` keeps only `best_checkpoint.pt` and `final_checkpoint.pt`.
  With AdamW this avoids writing ~10 large periodic checkpoints.
- Validation always runs on the full 10k validation split. Qualitative
  generation is disabled (`--num_qualitative_samples 0`) because it does not
  affect perplexity.
- The defaults target a GB10 (121 GB unified memory, CUDA 13). On other GPUs,
  set `BATCH_SIZE`/`GRAD_ACCUM` so that their product remains 48.
