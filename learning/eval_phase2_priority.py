#!/usr/bin/env python3
"""
Priorité post–Phase 2 :
  1) baseline random (même shape que projected)
  2) continuation (visuel+instr+préfixe GT → suite)
  3) mini-QA (1 question simple)

Usage :
  python learning/eval_phase2_priority.py \
    --data-root data/test \
    --phase2-ckpt checkpoints/optical_adapter_phase2/adapter_phase2_final.pt \
    --max-samples 8
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.optical_adapter import OpticalAdapter

DTYPE = torch.bfloat16
INSTRUCTION = (
    "Convertis strictement le document visuel ci-dessus en texte. "
    "Reproduis le contenu de manière fidèle et complète, sans commentaire."
)


def load_tokens(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for k in ("tokens", "visual_tokens", "causal_tokens_896"):
            if k in obj:
                t = obj[k]
                break
        else:
            t = next(iter(obj.values()))
    else:
        t = obj
    if t.ndim == 3:
        t = t.squeeze(0)
    return t.float()


def load_adapter(ckpt: Path, device) -> OpticalAdapter:
    m = OpticalAdapter(896, 2048).to(device).to(DTYPE)
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = raw["adapter"] if isinstance(raw, dict) and "adapter" in raw else raw
    m.load_state_dict(state)
    m.eval()
    return m


@torch.no_grad()
def project(adapter, tokens, device):
    return adapter(tokens.unsqueeze(0).to(device, DTYPE))  # [1, N, 2048]


@torch.no_grad()
def generate_from_prefix(qwen, tokenizer, prefix, max_new_tokens=256):
    device = prefix.device
    attn = torch.ones(1, prefix.shape[1], device=device, dtype=torch.long)
    gen = qwen.generate(
        inputs_embeds=prefix,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(gen[0], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/test")
    ap.add_argument("--qwen-path", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--phase2-ckpt", default="checkpoints/optical_adapter_phase2/adapter_phase2_final.pt")
    ap.add_argument("--max-samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gt-prefix-tokens", type=int, default=40, help="nb de tokens GT pour continuation")
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.data_root)
    pairs = []
    for pt in sorted((root / "tokens_896").glob("*.pt")):
        txt = root / "texts" / f"{pt.stem}.txt"
        if txt.exists():
            pairs.append((pt, txt))
    random.shuffle(pairs)
    pairs = pairs[: args.max_samples]
    print(f">>> {len(pairs)} samples")

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

    tok = AutoTokenizer.from_pretrained(args.qwen_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = qwen.model.embed_tokens.weight.device
    adapter = load_adapter(Path(args.phase2_ckpt), device)
    instr_ids = tok(INSTRUCTION, add_special_tokens=True, return_tensors="pt")["input_ids"].squeeze(0)

    sep = "=" * 72
    print(sep)
    print("OptiZip — Priority eval (random / continuation / QA)")
    print(sep)

    for pt, txt in pairs:
        tokens = load_tokens(pt)
        text = txt.read_text(encoding="utf-8").strip()
        gt_ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
        n_pref = min(args.gt_prefix_tokens, max(1, gt_ids.numel() // 4))

        print(f"\n{'-'*72}\nSAMPLE {pt.stem}  Nvis={tokens.shape[0]}  Ngt={gt_ids.numel()}")
        print(f"GT head: {text[:120]!r}")

        # --- projected réel ---
        proj = project(adapter, tokens, device)  # [1,N,2048]
        instr_emb = qwen.model.embed_tokens(instr_ids.unsqueeze(0).to(device))

        # 1) FREE GEN — adapter
        prefix_real = torch.cat([proj, instr_emb], dim=1)
        gen_real = generate_from_prefix(qwen, tok, prefix_real)
        print(f"\n[1] FREE GEN (adapter)\n{gen_real[:220]!r}")

        # 1b) FREE GEN — random (même shape, stats proches)
        rnd = torch.randn_like(proj) * 0.25  # std ~ Phase2
        prefix_rnd = torch.cat([rnd, instr_emb], dim=1)
        gen_rnd = generate_from_prefix(qwen, tok, prefix_rnd)
        print(f"\n[1b] FREE GEN (random embeds)\n{gen_rnd[:220]!r}")

        # 2) CONTINUATION — visuel+instr+40 tokens GT
        gt_pref_ids = gt_ids[:n_pref].unsqueeze(0).to(device)
        gt_pref_emb = qwen.model.embed_tokens(gt_pref_ids)
        prefix_cont = torch.cat([proj, instr_emb, gt_pref_emb], dim=1)
        gen_cont = generate_from_prefix(qwen, tok, prefix_cont, max_new_tokens=120)
        gt_tail = tok.decode(gt_ids[n_pref : n_pref + 80], skip_special_tokens=True)
        print(f"\n[2] CONTINUATION (après {n_pref} tokens GT)")
        print(f"    GT suite : {gt_tail[:180]!r}")
        print(f"    GEN suite: {gen_cont[:180]!r}")

        # 3) Mini-QA
        q = (
            "Réponds en une courte phrase : quelle est la première ligne ou la "
            "première déclaration visible dans ce document ?"
        )
        q_ids = tok(q, add_special_tokens=True, return_tensors="pt")["input_ids"].squeeze(0)
        q_emb = qwen.model.embed_tokens(q_ids.unsqueeze(0).to(device))
        prefix_qa = torch.cat([proj, q_emb], dim=1)
        gen_qa = generate_from_prefix(qwen, tok, prefix_qa, max_new_tokens=80)
        print(f"\n[3] MINI-QA\n    Q: {q}")
        print(f"    A: {gen_qa[:200]!r}")

    print(f"\n{sep}\nFin priority eval\n{sep}")


if __name__ == "__main__":
    main()
