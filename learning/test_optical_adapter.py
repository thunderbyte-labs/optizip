#!/usr/bin/env python3
"""
Validation sur data_test/ + détection hard examples (version corrigée)
"""
import os
import random
import shutil
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ================== CONFIG ==================
QWEN_PATH = "Qwen/Qwen3.6-35B-A3B"
ADAPTER_CKPT = "checkpoints/optical_adapter_phase1/adapter_final.pt"

TEST_ROOT = Path("data_test")
TRAIN_ROOT = Path("data_training")

NUM_SAMPLES = 200
MAX_TEXT_LEN = 256
LOSS_THRESHOLD = 2.0
SEED = 42
MOVE_HARD_EXAMPLES = True

dtype = torch.bfloat16
# ============================================

class OpticalAdapter(torch.nn.Module):
    def __init__(self, in_dim=896, out_dim=2048):
        super().__init__()
        self.norm = torch.nn.RMSNorm(in_dim)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(out_dim, out_dim, bias=False),
        )
    def forward(self, x):
        return self.proj(self.norm(x))

def compute_loss(qwen, adapter, embed_device, visual_tokens, input_ids):
    visual = visual_tokens.unsqueeze(0).to(embed_device, dtype)
    input_ids = input_ids.unsqueeze(0).to(embed_device)
    with torch.no_grad():
        projected = adapter(visual)
        text_embeds = qwen.model.embed_tokens(input_ids)
        inputs_embeds = torch.cat([projected, text_embeds], dim=1)
        vis_len = projected.shape[1]
        attention_mask = torch.ones(1, inputs_embeds.shape[1], device=embed_device)
        labels = input_ids.clone()
        vis_labels = torch.full((1, vis_len), -100, device=embed_device, dtype=torch.long)
        labels = torch.cat([vis_labels, labels], dim=1)
        outputs = qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, use_cache=False)
    return outputs.loss.item()

def main():
    random.seed(SEED)
    print(">>> Loading Qwen (4-bit) + Adapter...")
    qwen = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        trust_remote_code=True, attn_implementation="eager",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"),
        local_files_only=True
    ).eval()
    embed_device = qwen.model.embed_tokens.weight.device

    adapter = OpticalAdapter().to(embed_device).to(torch.bfloat16)
    adapter.load_state_dict(torch.load(ADAPTER_CKPT, map_location="cpu")["adapter"])
    adapter.eval()

    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    tokens_dir = TEST_ROOT / "tokens_896"
    texts_dir = TEST_ROOT / "texts"

    available = []
    seen = set()
    for pt_path in tokens_dir.glob("*.pt"):
        if pt_path.stem in seen: continue
        seen.add(pt_path.stem)
        txt_path = texts_dir / (pt_path.stem + ".txt")
        if txt_path.exists():
            available.append((pt_path, txt_path))

    print(f">>> {len(available)} fichiers uniques disponibles dans data_test/")
    selected = random.sample(available, min(NUM_SAMPLES, len(available)))

    results = []
    for pt_path, txt_path in tqdm(selected, desc="Validation"):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        visual_tokens = data["tokens"]
        text = txt_path.read_text(encoding="utf-8").strip()
        tokenized = tokenizer(text, max_length=MAX_TEXT_LEN, truncation=True, return_tensors="pt")
        input_ids = tokenized["input_ids"].squeeze(0)
        loss = compute_loss(qwen, adapter, embed_device, visual_tokens, input_ids)
        results.append({"name": pt_path.stem, "pt_path": pt_path, "txt_path": txt_path,
                        "loss": loss, "n_tokens": visual_tokens.shape[0]})

    losses = [r["loss"] for r in results]
    sorted_results = sorted(results, key=lambda x: x["loss"], reverse=True)

    print("\n" + "="*65)
    print(f"STATISTIQUES SUR {len(results)} ÉCHANTILLONS (data_test)")
    print(f"Mean loss   : {sum(losses)/len(losses):.4f}")
    print(f"Median loss : {sorted(losses)[len(losses)//2]:.4f}")
    print(f"Min / Max   : {min(losses):.4f} / {max(losses):.4f}")
    print(f"Threshold   : {LOSS_THRESHOLD}")
    print("="*65)

    print("\n>>> Top 10 pires (uniques) :")
    seen_names = set()
    for r in sorted_results:
        if r["name"] in seen_names: continue
        seen_names.add(r["name"])
        print(f"  {r['name']:<25} | loss={r['loss']:.4f} | N={r['n_tokens']}")
        if len(seen_names) >= 10: break

    hard = [r for r in results if r["loss"] > LOSS_THRESHOLD]
    print(f"\n>>> {len(hard)} échantillons avec loss > {LOSS_THRESHOLD}")

    if MOVE_HARD_EXAMPLES and hard:
        print(">>> Transfert des hard examples...")
        for r in hard:
            (TRAIN_ROOT / "tokens_896").mkdir(parents=True, exist_ok=True)
            (TRAIN_ROOT / "texts").mkdir(parents=True, exist_ok=True)
            shutil.copy2(r["pt_path"], TRAIN_ROOT / "tokens_896" / r["pt_path"].name)
            shutil.copy2(r["txt_path"], TRAIN_ROOT / "texts" / r["txt_path"].name)
        print(f"    {len(hard)} fichiers transférés.")

    print("\n✅ Terminé.")

if __name__ == "__main__":
    main()
