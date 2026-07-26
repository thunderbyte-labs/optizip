#!/usr/bin/env python3
"""
Phase 1 - Alignment de l'OpticalAdapter
Utilise les tokens 896-dim déjà pré-calculés (pas de DeepSeek)
"""

import os
import math
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    get_cosine_schedule_with_warmup, BitsAndBytesConfig
)
from tqdm import tqdm
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = "cuda"
dtype = torch.bfloat16

# -------------------------------------------------------
# Optical Adapter
# -------------------------------------------------------
class OpticalAdapter(nn.Module):
    def __init__(self, in_dim=896, out_dim=2048):
        super().__init__()
        self.norm = nn.RMSNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim, bias=False),
        )

    def forward(self, x):
        return self.proj(self.norm(x))

# -------------------------------------------------------
# Dataset (tokens pré-calculés + texte)
# -------------------------------------------------------
class OpticalAlignmentDataset(Dataset):
    def __init__(self, root: str, tokenizer, max_text_len=2048):
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.samples = []

        tokens_dir = self.root / "tokens_896"
        texts_dir  = self.root / "texts"

        for pt_path in sorted(tokens_dir.glob("*.pt")):
            txt_path = texts_dir / (pt_path.stem + ".txt")
            if txt_path.exists():
                self.samples.append((pt_path, txt_path))

        print(f"Dataset prêt : {len(self.samples)} paires tokens + texte")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt_path, txt_path = self.samples[idx]

        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        tokens = data["tokens"]  # [N, 896]

        text = txt_path.read_text(encoding="utf-8").strip()

        tokenized = self.tokenizer(
            text,
            max_length=self.max_text_len,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )

        return {
            "visual_tokens": tokens,                     # [N, 896]
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
        }

def collate_fn(batch):
    return batch[0]  # batch_size = 1

# -------------------------------------------------------
# Training
# -------------------------------------------------------
def train():
    # ========== CONFIG ==========
    DATA_ROOT = "data_training"
    QWEN_PATH = "Qwen/Qwen3.6-35B-A3B"
    OUTPUT_DIR = Path("checkpoints/optical_adapter_phase1")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 1
    GRAD_ACCUM = 16
    LR = 1e-4
    NUM_EPOCHS = 1
    WARMUP_RATIO = 0.03
    MAX_TEXT_LEN = 256
    SAVE_EVERY = 100

    # ========== Models ==========
    print(">>> Loading Qwen3.6-35B-A3B (frozen, 4-bit)...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,          # bfloat16
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        #llm_int8_enable_fp32_cpu_offload=True,
    )
    
    qwen = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH,
        dtype=dtype,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        attn_implementation="eager",
        quantization_config=quantization_config,
        #low_cpu_mem_usage=True,
    ).eval()
    
    for p in qwen.parameters():
        p.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === Device réel de l'embedding (solution robuste pour MoE custom) ===
    embed_device = qwen.model.embed_tokens.weight.device
    print(f">>> embed_tokens device détecté : {embed_device}")

    print(">>> Creating OpticalAdapter (trainable)...")
    adapter = OpticalAdapter(896, 2048).to(embed_device).to(dtype)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=0.01)

    # ========== Data ==========
    dataset = OpticalAlignmentDataset(DATA_ROOT, tokenizer, max_text_len=MAX_TEXT_LEN)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    total_steps = math.ceil(len(loader) / GRAD_ACCUM) * NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps
    )

    # ========== Loop ==========
    print(f"\n>>> Starting Phase 1 | {len(dataset)} samples | {total_steps} steps")
    global_step = 0
    adapter.train()

    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        optimizer.zero_grad()

        for step, sample in enumerate(pbar):
            # Tout sur le device réel de l'embedding
            visual = sample["visual_tokens"].unsqueeze(0).to(embed_device, dtype)   # [1, N, 896]
            input_ids = sample["input_ids"].unsqueeze(0).to(embed_device)
            labels = input_ids.clone()

            # Project
            projected = adapter(visual)                                             # [1, N, 2048]

            # Text embeds
            text_embeds = qwen.model.embed_tokens(input_ids)                        # [1, L, 2048]

            # Concat
            inputs_embeds = torch.cat([projected, text_embeds], dim=1)

            # Masks & labels (tous sur embed_device)
            vis_len = projected.shape[1]
            vis_mask = torch.ones(1, vis_len, device=embed_device, dtype=torch.long)
            txt_mask = sample["attention_mask"].unsqueeze(0).to(embed_device)
            attention_mask = torch.cat([vis_mask, txt_mask], dim=1)

            vis_labels = torch.full((1, vis_len), -100, device=embed_device, dtype=torch.long)
            labels = torch.cat([vis_labels, labels], dim=1)
            # Forward
            outputs = qwen(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                pbar.set_postfix({
                    "loss": f"{loss.item()*GRAD_ACCUM:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "N": vis_len
                })

                if global_step % SAVE_EVERY == 0:
                    ckpt = OUTPUT_DIR / f"adapter_step_{global_step:05d}.pt"
                    torch.save({
                        "adapter": adapter.state_dict(),
                        "step": global_step,
                        "loss": loss.item() * GRAD_ACCUM
                    }, ckpt)
                    print(f"\n  → Saved {ckpt}")

        # End of epoch
        torch.save({
            "adapter": adapter.state_dict(),
            "epoch": epoch + 1
        }, OUTPUT_DIR / f"adapter_epoch_{epoch+1}.pt")

    # Final
    torch.save({"adapter": adapter.state_dict()}, OUTPUT_DIR / "adapter_final.pt")
    print("\n✅ Phase 1 terminée avec succès")
    print(f"   Checkpoints : {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
