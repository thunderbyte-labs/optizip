#!/usr/bin/env python3
"""
Phase 2 – Transmission visuel → texte (Qwen gelé)

Entrée  : tokens visuels projetés + instruction
Cible   : texte ground truth (teacher forcing)
Perte   : entropie croisée de Qwen (labels -100 sur préfixe)
Init    : poids Phase 1 (adapter_final.pt)

Usage sample 50 points :
  python learning/train_optical_adapter_phase2.py --max-samples 50

Usage nuit (~14 h) :
  python learning/train_optical_adapter_phase2.py --max-samples 0
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

from models.optical_adapter import OpticalAdapter

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEVICE = "cuda"
DTYPE = torch.bfloat16

INSTRUCTION = (
    "Convertis strictement le document visuel ci-dessus en texte. "
    "Reproduis le contenu de manière fidèle et complète, sans commentaire."
)


class Phase2Dataset(Dataset):
    def __init__(self, root: str, tokenizer, max_text_len: int = 1024, max_samples: int = 0):
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.samples = []

        tokens_dir = self.root / "tokens_896"
        texts_dir = self.root / "texts"

        for pt_path in sorted(tokens_dir.glob("*.pt")):
            txt_path = texts_dir / f"{pt_path.stem}.txt"
            if txt_path.exists():
                self.samples.append((pt_path, txt_path))

        if max_samples and max_samples > 0:
            self.samples = self.samples[:max_samples]

        print(f"Dataset Phase 2 : {len(self.samples)} paires (root={root})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt_path, txt_path = self.samples[idx]
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        tokens = data["tokens"] if isinstance(data, dict) else data  # [N, 896]
        text = txt_path.read_text(encoding="utf-8").strip()

        instr = self.tokenizer(
            INSTRUCTION,
            add_special_tokens=True,
            return_tensors="pt",
        )
        target = self.tokenizer(
            text,
            max_length=self.max_text_len,
            truncation=True,
            padding=False,
            add_special_tokens=False,
            return_tensors="pt",
        )

        return {
            "visual_tokens": tokens.float(),
            "instr_ids": instr["input_ids"].squeeze(0),
            "target_ids": target["input_ids"].squeeze(0),
            "name": pt_path.stem,
            "n_visual": tokens.shape[0],
            "n_target": target["input_ids"].shape[1],
        }


def collate_fn(batch):
    return batch[0]  # batch_size = 1


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(">>> Chargement Qwen (gelé, 4-bit)...")
    qwen = AutoModelForCausalLM.from_pretrained(
        args.qwen_path,
        dtype=DTYPE,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        attn_implementation="eager",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    ).eval()
    for p in qwen.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embed_device = qwen.model.embed_tokens.weight.device
    print(f">>> embed_tokens device : {embed_device}")

    print(">>> OpticalAdapter (init Phase 1 si dispo)...")
    adapter = OpticalAdapter(896, 2048).to(embed_device).to(DTYPE)
    if Path(args.init_ckpt).exists():
        raw = torch.load(args.init_ckpt, map_location="cpu", weights_only=True)
        state = raw["adapter"] if isinstance(raw, dict) and "adapter" in raw else raw
        adapter.load_state_dict(state)
        print(f"    init depuis {args.init_ckpt}")
    else:
        print(f"    ATTENTION : {args.init_ckpt} introuvable, init aléatoire")

    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)

    dataset = Phase2Dataset(
        args.data_root, tokenizer, max_text_len=args.max_text_len, max_samples=args.max_samples
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Aucune paire dans {args.data_root}")

    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.03)),
        num_training_steps=max(1, total_steps),
    )

    print(f"\n>>> Phase 2 | samples={len(dataset)} | epochs={args.epochs} | "
          f"grad_accum={args.grad_accum} | steps≈{total_steps}")
    print(f"    instruction: {INSTRUCTION[:60]}...")

    global_step = 0
    t0 = time.perf_counter()
    samples_seen = 0
    loss_window = []

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)

        for step, sample in enumerate(pbar):
            t_step = time.perf_counter()

            visual = sample["visual_tokens"].unsqueeze(0).to(embed_device, DTYPE)
            instr_ids = sample["instr_ids"].unsqueeze(0).to(embed_device)
            target_ids = sample["target_ids"].unsqueeze(0).to(embed_device)

            projected = adapter(visual)  # [1, N, 2048]
            instr_embeds = qwen.model.embed_tokens(instr_ids)
            text_embeds = qwen.model.embed_tokens(target_ids)

            # Préfixe produit : visuel + instruction  |  teacher forcing : + texte
            inputs_embeds = torch.cat([projected, instr_embeds, text_embeds], dim=1)

            vis_len = projected.shape[1]
            instr_len = instr_embeds.shape[1]
            txt_len = text_embeds.shape[1]
            prefix_len = vis_len + instr_len

            attention_mask = torch.ones(
                1, inputs_embeds.shape[1], device=embed_device, dtype=torch.long
            )
            labels = torch.cat(
                [
                    torch.full((1, prefix_len), -100, device=embed_device, dtype=torch.long),
                    target_ids,
                ],
                dim=1,
            )

            outputs = qwen(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = outputs.loss / args.grad_accum
            loss.backward()

            samples_seen += 1
            loss_val = loss.item() * args.grad_accum
            loss_window.append(loss_val)
            if len(loss_window) > 20:
                loss_window.pop(0)

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.save_every == 0:
                    ckpt = output_dir / f"adapter_phase2_step_{global_step:05d}.pt"
                    torch.save(
                        {
                            "adapter": adapter.state_dict(),
                            "step": global_step,
                            "loss": loss_val,
                            "samples_seen": samples_seen,
                        },
                        ckpt,
                    )

            dt = time.perf_counter() - t_step
            elapsed = time.perf_counter() - t0
            sps = samples_seen / max(elapsed, 1e-6)  # samples / s
            mean_loss = sum(loss_window) / len(loss_window)

            # Extrapolation : temps pour 1 epoch complète sur ce dataset, et pour N samples
            sec_per_sample = elapsed / max(samples_seen, 1)
            eta_50 = sec_per_sample * 50
            eta_full_epoch = sec_per_sample * len(dataset)
            eta_10k = sec_per_sample * 10_000

            pbar.set_postfix({
                "loss": f"{mean_loss:.4f}",
                "s/sample": f"{sec_per_sample:.2f}",
                "sps": f"{sps:.3f}",
                "Nvis": sample["n_visual"],
                "Ntgt": sample["n_target"],
            })

            # Log détaillé périodique
            if samples_seen <= 3 or samples_seen % 10 == 0:
                print(
                    f"\n[log] sample={samples_seen} name={sample['name']} "
                    f"loss={loss_val:.4f} mean20={mean_loss:.4f} "
                    f"sec/sample={sec_per_sample:.2f} "
                    f"ETA@50={eta_50/60:.1f}min ETA@epoch={eta_full_epoch/3600:.2f}h "
                    f"ETA@10k={eta_10k/3600:.2f}h "
                    f"vis={vis_len} instr={instr_len} tgt={txt_len}"
                )

        # fin d'epoch
        torch.save(
            {"adapter": adapter.state_dict(), "epoch": epoch + 1, "samples_seen": samples_seen},
            output_dir / f"adapter_phase2_epoch_{epoch+1}.pt",
        )

    torch.save({"adapter": adapter.state_dict()}, output_dir / "adapter_phase2_final.pt")
    total = time.perf_counter() - t0
    print("\n✅ Phase 2 terminée")
    print(f"   samples={samples_seen} | durée={total/60:.1f} min | "
          f"sec/sample={total/max(samples_seen,1):.2f}")
    print(f"   checkpoints : {output_dir}")
    print(
        f"   Extrapolation linéaire : "
        f"1000 samples ≈ {(total/max(samples_seen,1))*1000/3600:.2f} h | "
        f"10000 samples ≈ {(total/max(samples_seen,1))*10000/3600:.2f} h"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data_training")
    p.add_argument("--qwen-path", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--init-ckpt", default="checkpoints/optical_adapter_phase1/adapter_final.pt")
    p.add_argument("--output-dir", default="checkpoints/optical_adapter_phase2")
    p.add_argument("--max-samples", type=int, default=50, help="0 = tout le dataset")
    p.add_argument("--max-text-len", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--save-every", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
