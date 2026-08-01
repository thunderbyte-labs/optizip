#!/usr/bin/env python3
"""
Sanity check Phase 2 – étapes B / C / D
  B : stats des embeddings projetés (Phase 1 vs Phase 2)
  C : loss hold-out (protocole Phase 2, teacher forcing)
  D : génération libre vs ground truth

Usage :
  python learning/test_optical_adapter_phase2.py \
    --data-root data/test \
    --max-samples 50 \
    --phase2-ckpt checkpoints/optical_adapter_phase2/adapter_phase2_final.pt \
    --phase1-ckpt checkpoints/optical_adapter_phase1/adapter_final.pt
"""

from __future__ import annotations

import argparse
import random
import sys
import time
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
        for k in ("tokens", "visual_tokens", "causal_tokens_896", "embeddings"):
            if k in obj:
                t = obj[k]
                break
        else:
            t = next(iter(obj.values()))
    else:
        t = obj
    if t.ndim == 3:
        t = t.squeeze(0)
    assert t.ndim == 2 and t.shape[1] == 896, f"Shape inattendue: {tuple(t.shape)} ({path})"
    return t.float()


def load_adapter(ckpt: Path, device, dtype) -> OpticalAdapter:
    m = OpticalAdapter(896, 2048).to(device).to(dtype)
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = raw["adapter"] if isinstance(raw, dict) and "adapter" in raw else raw
    m.load_state_dict(state)
    m.eval()
    return m


@torch.no_grad()
def project(adapter, tokens: torch.Tensor, device) -> torch.Tensor:
    x = tokens.unsqueeze(0).to(device, DTYPE)
    return adapter(x).squeeze(0).float().cpu()  # [N, 2048]


@torch.no_grad()
def phase2_loss(qwen, adapter, embed_device, tokens, instr_ids, target_ids) -> float:
    visual = tokens.unsqueeze(0).to(embed_device, DTYPE)
    instr_ids = instr_ids.unsqueeze(0).to(embed_device)
    target_ids = target_ids.unsqueeze(0).to(embed_device)

    projected = adapter(visual)
    instr_emb = qwen.model.embed_tokens(instr_ids)
    text_emb = qwen.model.embed_tokens(target_ids)
    inputs_embeds = torch.cat([projected, instr_emb, text_emb], dim=1)

    prefix_len = projected.shape[1] + instr_emb.shape[1]
    labels = torch.cat(
        [
            torch.full((1, prefix_len), -100, device=embed_device, dtype=torch.long),
            target_ids,
        ],
        dim=1,
    )
    attn = torch.ones(1, inputs_embeds.shape[1], device=embed_device, dtype=torch.long)
    out = qwen(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        labels=labels,
        use_cache=False,
    )
    return float(out.loss.item())


@torch.no_grad()
def free_generate(qwen, tokenizer, adapter, embed_device, tokens, instr_ids, max_new_tokens=512) -> str:
    visual = tokens.unsqueeze(0).to(embed_device, DTYPE)
    instr_ids_b = instr_ids.unsqueeze(0).to(embed_device)

    projected = adapter(visual)
    instr_emb = qwen.model.embed_tokens(instr_ids_b)
    prefix = torch.cat([projected, instr_emb], dim=1)
    attn = torch.ones(1, prefix.shape[1], device=embed_device, dtype=torch.long)

    gen = qwen.generate(
        inputs_embeds=prefix,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # generate avec inputs_embeds renvoie souvent seulement les nouveaux tokens
    # ou prefix_len + new selon version ; on décode tout et on nettoie
    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    return text.strip()


def ascii_bar(value: float, lo: float, hi: float, width: int = 24) -> str:
    if hi <= lo:
        return "[" + " " * width + "]"
    x = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    n = int(round(x * width))
    return "[" + "#" * n + "-" * (width - n) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/test")
    ap.add_argument("--qwen-path", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--phase2-ckpt", default="checkpoints/optical_adapter_phase2/adapter_phase2_final.pt")
    ap.add_argument("--phase1-ckpt", default="checkpoints/optical_adapter_phase1/adapter_final.pt")
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--max-text-len", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--gen-samples", type=int, default=8, help="nb d'exemples affichés en génération libre")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    data_root = Path(args.data_root)
    tokens_dir = data_root / "tokens_896"
    texts_dir = data_root / "texts"

    pairs = []
    for pt in sorted(tokens_dir.glob("*.pt")):
        txt = texts_dir / f"{pt.stem}.txt"
        if txt.exists():
            pairs.append((pt, txt))
    if not pairs:
        raise SystemExit(f"Aucune paire tokens/texte dans {data_root}")

    random.shuffle(pairs)
    pairs = pairs[: args.max_samples]
    print(f">>> {len(pairs)} samples hold-out depuis {data_root}")

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
    print(f">>> embed device = {embed_device}")

    adapter2 = load_adapter(Path(args.phase2_ckpt), embed_device, DTYPE)
    adapter1 = None
    if Path(args.phase1_ckpt).exists():
        adapter1 = load_adapter(Path(args.phase1_ckpt), embed_device, DTYPE)
        print(f">>> Phase 1 chargée : {args.phase1_ckpt}")
    else:
        print(">>> Phase 1 absente — stats comparatives sautées")

    instr = tokenizer(INSTRUCTION, add_special_tokens=True, return_tensors="pt")
    instr_ids = instr["input_ids"].squeeze(0)

    # ------------------------------------------------------------------
    # B + C
    # ------------------------------------------------------------------
    rows = []
    t0 = time.perf_counter()
    for i, (pt, txt) in enumerate(pairs, 1):
        tokens = load_tokens(pt)
        text = txt.read_text(encoding="utf-8").strip()
        target = tokenizer(
            text,
            max_length=args.max_text_len,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        proj2 = project(adapter2, tokens, embed_device)
        stats2 = (
            float(proj2.mean()),
            float(proj2.std()),
            float(proj2.norm(dim=-1).mean()),
        )
        stats1 = None
        if adapter1 is not None:
            proj1 = project(adapter1, tokens, embed_device)
            stats1 = (
                float(proj1.mean()),
                float(proj1.std()),
                float(proj1.norm(dim=-1).mean()),
            )

        loss2 = phase2_loss(qwen, adapter2, embed_device, tokens, instr_ids, target)
        loss1 = None
        if adapter1 is not None:
            loss1 = phase2_loss(qwen, adapter1, embed_device, tokens, instr_ids, target)

        rows.append(
            {
                "name": pt.stem,
                "n_vis": tokens.shape[0],
                "n_tgt": int(target.numel()),
                "stats2": stats2,
                "stats1": stats1,
                "loss2": loss2,
                "loss1": loss1,
                "pt": pt,
                "txt": txt,
                "tokens": tokens,
                "target_ids": target,
                "text": text,
            }
        )
        if i % 10 == 0 or i == 1:
            print(f"  [{i}/{len(pairs)}] {pt.stem} loss2={loss2:.4f} Nvis={tokens.shape[0]}")

    elapsed_bc = time.perf_counter() - t0
    losses2 = [r["loss2"] for r in rows]
    losses1 = [r["loss1"] for r in rows if r["loss1"] is not None]

    def pct(xs, p):
        ys = sorted(xs)
        if not ys:
            return float("nan")
        k = min(len(ys) - 1, max(0, int(round((p / 100) * (len(ys) - 1)))))
        return ys[k]

    # ------------------------------------------------------------------
    # D – génération libre sur un sous-ensemble
    # ------------------------------------------------------------------
    gen_rows = sorted(rows, key=lambda r: r["loss2"])  # bons + mauvais
    pick = []
    # 3 meilleurs, 3 médians, 2 pires
    n = len(gen_rows)
    idxs = sorted(set(
        [0, 1, 2, n // 2 - 1, n // 2, n // 2 + 1, n - 2, n - 1]
    ))
    idxs = [i for i in idxs if 0 <= i < n][: args.gen_samples]
    for i in idxs:
        pick.append(gen_rows[i])

    print(f"\n>>> Génération libre sur {len(pick)} samples...")
    gen_results = []
    t1 = time.perf_counter()
    for r in pick:
        try:
            out = free_generate(
                qwen, tokenizer, adapter2, embed_device,
                r["tokens"], instr_ids, max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            out = f"[ERREUR generate] {e}"
        gen_results.append((r, out))
        print(f"  gen {r['name']} (loss2={r['loss2']:.3f}) → {len(out)} chars")
    elapsed_d = time.perf_counter() - t1

    # ------------------------------------------------------------------
    # Rapport ASCII
    # ------------------------------------------------------------------
    means2 = [r["stats2"][0] for r in rows]
    stds2 = [r["stats2"][1] for r in rows]
    norms2 = [r["stats2"][2] for r in rows]

    sep = "=" * 72
    thin = "-" * 72
    lines = []
    lines.append(sep)
    lines.append("OptiZip — Sanity Report Phase 2")
    lines.append(sep)
    lines.append(f"data_root     : {data_root}")
    lines.append(f"phase2_ckpt   : {args.phase2_ckpt}")
    lines.append(f"phase1_ckpt   : {args.phase1_ckpt if adapter1 else '(absent)'}")
    lines.append(f"n_samples     : {len(rows)}")
    lines.append(f"temps B+C     : {elapsed_bc/60:.1f} min")
    lines.append(f"temps D       : {elapsed_d/60:.1f} min")
    lines.append(thin)

    lines.append("B) STATS EMBEDDINGS PROJETÉS (Phase 2)")
    lines.append(
        f"  mean  : {sum(means2)/len(means2):+.4f}   "
        f"[{min(means2):+.4f} .. {max(means2):+.4f}]"
    )
    lines.append(
        f"  std   : {sum(stds2)/len(stds2):.4f}   "
        f"[{min(stds2):.4f} .. {max(stds2):.4f}]"
    )
    lines.append(
        f"  norm  : {sum(norms2)/len(norms2):.4f}   "
        f"[{min(norms2):.4f} .. {max(norms2):.4f}]"
    )
    if adapter1 is not None:
        means1 = [r["stats1"][0] for r in rows]
        stds1 = [r["stats1"][1] for r in rows]
        norms1 = [r["stats1"][2] for r in rows]
        lines.append("  --- Phase 1 (même samples) ---")
        lines.append(f"  mean  : {sum(means1)/len(means1):+.4f}")
        lines.append(f"  std   : {sum(stds1)/len(stds1):.4f}")
        lines.append(f"  norm  : {sum(norms1)/len(norms1):.4f}")
    lines.append(thin)

    lines.append("C) LOSS HOLD-OUT — protocole Phase 2 (visuel+instruction → texte)")
    lines.append(f"  Phase2  mean={sum(losses2)/len(losses2):.4f}  "
                 f"median={pct(losses2,50):.4f}  "
                 f"p90={pct(losses2,90):.4f}  "
                 f"min={min(losses2):.4f}  max={max(losses2):.4f}")
    lines.append(f"  dist   {ascii_bar(sum(losses2)/len(losses2), 0.0, 3.0)}  (0 → 3)")
    if losses1:
        lines.append(f"  Phase1  mean={sum(losses1)/len(losses1):.4f}  "
                     f"median={pct(losses1,50):.4f}  "
                     f"min={min(losses1):.4f}  max={max(losses1):.4f}")
        delta = sum(losses1)/len(losses1) - sum(losses2)/len(losses2)
        lines.append(f"  Δ (P1 - P2) = {delta:+.4f}   "
                     f"({'P2 meilleur' if delta > 0 else 'P1 meilleur ou égal'})")
    lines.append("")
    lines.append("  Top 5 pires (Phase 2):")
    for r in sorted(rows, key=lambda x: -x["loss2"])[:5]:
        lines.append(f"    {r['name']:<28} loss2={r['loss2']:.4f}  "
                     f"Nvis={r['n_vis']:<4} Ntgt={r['n_tgt']}")
    lines.append("  Top 5 meilleurs (Phase 2):")
    for r in sorted(rows, key=lambda x: x["loss2"])[:5]:
        lines.append(f"    {r['name']:<28} loss2={r['loss2']:.4f}  "
                     f"Nvis={r['n_vis']:<4} Ntgt={r['n_tgt']}")
    lines.append(thin)

    lines.append("D) GÉNÉRATION LIBRE (extrait)")
    for r, out in gen_results:
        gt = r["text"]
        lines.append(f"  sample : {r['name']}   loss2={r['loss2']:.4f}   "
                     f"Nvis={r['n_vis']} Ntgt={r['n_tgt']}")
        lines.append(f"  --- GT (200 chars) ---")
        lines.append("  " + gt[:200].replace("\n", "\\n"))
        lines.append(f"  --- GEN (200 chars) ---")
        lines.append("  " + (out[:200] if out else "(vide)").replace("\n", "\\n"))
        lines.append("")
    lines.append(thin)

    # Verdict grossier
    mean_l = sum(losses2) / len(losses2)
    if mean_l < 0.8:
        verdict = "OK — loss hold-out basse ; passer à l'inspection qualitative D + test vLLM"
    elif mean_l < 1.5:
        verdict = "MITIGÉ — loss correcte mais vérifier génération libre avant prod"
    else:
        verdict = "FAIBLE — loss haute ; ne pas pousser en prod sans diagnostic"
    lines.append(f"VERDICT : {verdict}")
    lines.append(sep)

    report = "\n".join(lines)
    print("\n" + report)

    out_path = Path("checkpoints/optical_adapter_phase2/sanity_report_phase2.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nRapport sauvé → {out_path}")


if __name__ == "__main__":
    main()
