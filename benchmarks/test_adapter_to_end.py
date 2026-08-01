#!/usr/bin/env python3
"""
OptiZip – End-to-end validation (OpticalAdapter + vLLM)
Utilise les tokens_896 pré-calculés (bypass DeepEncoder)
"""

import io
import base64
import sys
from pathlib import Path

import torch
import requests

# Racine du projet
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.optical_adapter import OpticalAdapter

# ============================================================
# Config
# ============================================================
SAMPLE_ID      = "code_00000002"

ADAPTER_CKPT   = ROOT / "checkpoints/optical_adapter_phase1/adapter_final.pt"
TOKENS_PATH    = ROOT / f"data/test/tokens_896/{SAMPLE_ID}.pt"
TEXT_PATH      = ROOT / f"data/test/texts/{SAMPLE_ID}.txt"

VLLM_URL       = "http://192.168.128.10:8000/v1/chat/completions"
MODEL_NAME     = "/root/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4"
DEVICE         = "cuda:0"

QUESTION = "Extrais et reproduis fidèlement tout le texte visible dans ce document."


def load_tokens(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        # formats possibles selon le script d’extraction
        for key in ("tokens", "visual_tokens", "embeddings", "x"):
            if key in obj:
                obj = obj[key]
                break
        else:
            obj = next(iter(obj.values()))
    t = obj.float()
    assert t.ndim == 2 and t.shape[1] == 896, f"Shape inattendue: {t.shape}"
    return t


def project(tokens: torch.Tensor) -> torch.Tensor:
    adapter = OpticalAdapter(in_dim=896, out_dim=2048)
    raw = torch.load(ADAPTER_CKPT, map_location="cpu", weights_only=True)
    state = raw["adapter"] if isinstance(raw, dict) and "adapter" in raw else raw
    adapter.load_state_dict(state)
    adapter.eval().to(DEVICE)

    with torch.no_grad():
        out = adapter(tokens.to(DEVICE)).to(torch.bfloat16).cpu()
    return out


def to_b64(tensor: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    print("=" * 60)
    print("OptiZip – test_end_to_end (Adapter + vLLM)")
    print("=" * 60)

    print(f"\n[1] Chargement {TOKENS_PATH.name}")
    tokens = load_tokens(TOKENS_PATH)
    print(f"    shape = {tuple(tokens.shape)}")

    print("\n[2] Projection OpticalAdapter...")
    projected = project(tokens)
    print(f"    → {projected.shape[0]} tokens × 2048")

    print(f"    mean={projected.float().mean():.4f}  std={projected.float().std():.4f}  "
      f"norm={projected.float().norm(dim=-1).mean():.4f}")

    b64 = to_b64(projected)

    print("\n[3] Appel vLLM...")
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu es un expert en extraction de texte depuis des documents. "
                    "Reproduis le texte de manière fidèle et complète."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "prompt_embeds", "data": b64},
                    {"type": "text", "text": f"\n\n{QUESTION}"},
                ],
            },
        ],
        "max_tokens": 2048,
        "temperature": 0.1,
    }

    resp = requests.post(VLLM_URL, json=payload, timeout=180)
    resp.raise_for_status()
    result = resp.json()

    message = result["choices"][0]["message"]
    answer = (
        message.get("content")
        or message.get("reasoning")
        or message.get("reasoning_content")
        or ""
    )

    print("\n" + "=" * 60)
    print("RÉPONSE DE QWEN")
    print("=" * 60)
    print(answer[:4000] if answer else "(vide)")

    if TEXT_PATH.exists():
        print("\n" + "=" * 60)
        print("GROUND TRUTH")
        print("=" * 60)
        gt = TEXT_PATH.read_text()
        print(gt[:2000])
        if len(gt) > 2000:
            print("\n... (tronqué)")

    print("\n✅ Terminé")


if __name__ == "__main__":
    main()
