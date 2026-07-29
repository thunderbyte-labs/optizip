#!/usr/bin/env python3
"""
Génère tokens_896 manquants pour data/test (DeepSeek-OCR-2 multi-crop).
Format sortie : {"tokens": FloatTensor [N, 896]}  compatible Phase 1/2.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from transformers import AutoModel

# --- monkey patch éventuel (comme tes scripts) ---
import transformers.models.llama.modeling_llama as modeling_llama
if not hasattr(modeling_llama, "LlamaFlashAttention2"):
    class LlamaFlashAttention2(modeling_llama.LlamaAttention):
        pass
    modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2

# Réutilise les fonctions de ton multicrop (copie minimale)
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=768, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed.append(resized_img.crop(box))
    return processed, target_aspect_ratio


def get_transform(size):
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


@torch.no_grad()
def extract_multicrop(model, image, device, dtype, base_size=1024, image_size=768, max_crops=6):
    image = image.convert("RGB")
    all_tokens = []

    transform_g = get_transform(base_size)
    global_img = ImageOps.pad(image, (base_size, base_size), color=(127, 127, 127))
    pixel_g = transform_g(global_img).unsqueeze(0).to(device, dtype)
    tok_g = model.model.qwen2_model(model.model.sam_model(pixel_g))  # [1, 256, 896]
    all_tokens.append(tok_g)

    if max(image.size) > image_size:
        locals_, _ = dynamic_preprocess(image, 1, max_crops, image_size, False)
        transform_l = get_transform(image_size)
        for crop in locals_:
            pixel_l = transform_l(crop).unsqueeze(0).to(device, dtype)
            tok_l = model.model.qwen2_model(model.model.sam_model(pixel_l))
            all_tokens.append(tok_l)

    if len(all_tokens) > 1:
        g = all_tokens.pop(0)
        all_tokens.append(g)

    return torch.cat(all_tokens, dim=1).squeeze(0).float().cpu()  # [N, 896]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="data/test/images")
    ap.add_argument("--texts-dir", default="data/test/texts")
    ap.add_argument("--out-dir", default="data/test/tokens_896")
    ap.add_argument("--weights", default="./DeepSeek-OCR-2-weights")
    ap.add_argument("--num", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    texts_dir = Path(args.texts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = {p.stem for p in out_dir.glob("*.pt")}
    candidates = []
    for img in images_dir.glob("*.png"):
        stem = img.stem
        if stem in existing:
            continue
        if not (texts_dir / f"{stem}.txt").exists():
            continue
        candidates.append(img)

    random.seed(args.seed)
    random.shuffle(candidates)
    candidates = candidates[: args.num]
    print(f"À extraire : {len(candidates)} (déjà présents ignorés)")

    dtype = torch.bfloat16
    device = args.device
    print(">>> Chargement DeepSeek-OCR-2...")
    model = AutoModel.from_pretrained(
        args.weights,
        trust_remote_code=True,
        torch_dtype=dtype,
        _attn_implementation="eager",
        device_map=device,
        local_files_only=True,
    ).eval()

    ok, fail = 0, 0
    for i, img_path in enumerate(candidates, 1):
        try:
            image = Image.open(img_path).convert("RGB")
            tokens = extract_multicrop(model, image, device, dtype)
            torch.save({"tokens": tokens}, out_dir / f"{img_path.stem}.pt")
            ok += 1
            if i % 20 == 0 or i == 1:
                print(f"[{i}/{len(candidates)}] {img_path.name} → {tokens.shape}")
        except Exception as e:
            fail += 1
            print(f"ERREUR {img_path.name}: {e}")

    print(f"\n✅ ok={ok} fail={fail} | total .pt maintenant : {len(list(out_dir.glob('*.pt')))}")


if __name__ == "__main__":
    main()
