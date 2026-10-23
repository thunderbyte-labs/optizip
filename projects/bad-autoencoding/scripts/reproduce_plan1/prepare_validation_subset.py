#!/usr/bin/env python3
"""Create a validation-only Arrow dataset with a minimal image subset.

Useful when a second machine only needs to validate a checkpoint: instead of
copying every training image (tens of GB), this writes the validation Arrow
split plus symlinks (or copies) for the validation images only.

The output layout is compatible with the repository's Arrow datasets:

    <output-dir>/
        val_arrow/
        val_arrow_smoke/          (optional)
        images/<vision-mode>/<id>.png

Example:
    uv run python scripts/reproduce_plan1/prepare_validation_subset.py \
        --val-arrow data/training/full/val_arrow \
        --images-dir data/training/full/images/tiny \
        --output-dir data/training/full/val_only \
        --vision-mode tiny \
        --limit 32 \
        --copy
"""

import argparse
import shutil
import sys
from pathlib import Path

from datasets import load_from_disk


def link_or_copy(source: Path, destination: Path, copy: bool) -> None:
    if destination.exists() or destination.is_symlink():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-arrow", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=None,
                        help="Directory containing <id>.png. Omit for text-only regimes.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vision-mode", default="tiny")
    parser.add_argument("--limit", type=int, default=None,
                        help="Keep only the first N validation samples.")
    parser.add_argument("--smoke-size", type=int, default=32,
                        help="Also write val_arrow_smoke with this many samples (0 disables).")
    parser.add_argument("--copy", action="store_true",
                        help="Copy images instead of symlinking them.")
    args = parser.parse_args()

    if not (args.val_arrow / "dataset_info.json").exists():
        print(f"ERROR: {args.val_arrow} is not an Arrow dataset", file=sys.stderr)
        return 1

    dataset = load_from_disk(str(args.val_arrow))
    total = len(dataset)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, total)))
    if len(dataset) == 0:
        print("ERROR: selected validation subset is empty", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    arrow_out = output_dir / "val_arrow"
    if arrow_out.exists():
        shutil.rmtree(arrow_out)
    dataset.save_to_disk(str(arrow_out))
    print(f"Wrote {len(dataset):,} samples to {arrow_out}")

    if args.smoke_size > 0:
        smoke = dataset.select(range(min(args.smoke_size, len(dataset))))
        smoke_out = output_dir / "val_arrow_smoke"
        if smoke_out.exists():
            shutil.rmtree(smoke_out)
        smoke.save_to_disk(str(smoke_out))
        print(f"Wrote {len(smoke):,} samples to {smoke_out}")

    if args.images_dir is not None:
        if not args.images_dir.exists():
            print(f"ERROR: images directory not found: {args.images_dir}", file=sys.stderr)
            return 1
        image_out = output_dir / "images" / args.vision_mode
        image_out.mkdir(parents=True, exist_ok=True)
        missing = 0
        for sample_id in dataset["id"]:
            source = args.images_dir / f"{sample_id}.png"
            if not source.exists():
                missing += 1
                continue
            link_or_copy(source, image_out / source.name, args.copy)
        print(
            f"Linked/copied {len(dataset) - missing:,} images to {image_out} "
            f"({missing:,} missing)"
        )
        if missing:
            print("WARNING: some images were missing; validation will fail for those samples", file=sys.stderr)
            return 2 if missing == len(dataset) else 0

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
