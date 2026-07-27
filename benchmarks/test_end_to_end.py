#!/usr/bin/env python3
"""
Test end-to-end OptiZip :
  Image → Preprocessing Microservice (Hera) → prompt_embeds → vLLM (Hephaistos) → réponse
"""

import base64
import json
from pathlib import Path
import requests

# ============================================================
# Config
# ============================================================
PREPROCESS_URL = "http://localhost:8001/extract_visual_context"
VLLM_URL = "http://192.168.128.10:8000/v1/chat/completions"
MODEL_NAME = "/root/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4"

# Image de test (change si besoin)
IMAGE_PATH = Path("data/test/images/code_00000000.png")
TEXT_PATH  = Path("data/test/texts/code_00000000.txt")   # ground truth

QUESTION = "Extrais et reproduis fidèlement tout le texte visible dans ce document."


def main():
    print("=" * 60)
    print("OptiZip E2E Test")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Envoi de l'image au microservice
    # ----------------------------------------------------------
    print(f"\n[1] Envoi de {IMAGE_PATH.name} au preprocessing...")

    with open(IMAGE_PATH, "rb") as f:
        files = {"images": (IMAGE_PATH.name, f, "image/png")}
        resp = requests.post(PREPROCESS_URL, files=files, timeout=120)

    if resp.status_code != 200:
        print("ERREUR preprocessing:", resp.status_code, resp.text)
        return

    data = resp.json()
    b64_embeds = data["prompt_embeds_b64"]
    n_tokens = data["num_visual_tokens"]
    timings = data.get("timings_ms", {})

    print(f"    → {n_tokens} tokens visuels générés")
    print(f"    → timings: {timings}")

    # ----------------------------------------------------------
    # 2. Appel à vLLM avec les prompt_embeds
    # ----------------------------------------------------------
    print(f"\n[2] Envoi des embeddings à vLLM...")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "Tu es un expert en extraction de texte depuis des documents. Reproduis le texte de manière fidèle et complète."
            },
            {
                "role": "user",
                "content": [
                    {"type": "prompt_embeds", "data": b64_embeds},
                    {"type": "text", "text": f"\n\n{QUESTION}"}
                ]
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.1,
    }

    resp = requests.post(VLLM_URL, json=payload, timeout=180)

    if resp.status_code != 200:
        print("ERREUR vLLM:", resp.status_code, resp.text)
        return

    result = resp.json()
    
    # Debug complet
    print("\n" + "=" * 60)
    print("RAW RESPONSE (debug)")
    print("=" * 60)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False)[:3000])
    
    # Puis extraction plus robuste
    message = result["choices"][0]["message"]
    answer = (
        message.get("content")
        or message.get("reasoning")
        or message.get("reasoning_content")
        or message.get("text")
        or str(message)
    )

    # ----------------------------------------------------------
    # 3. Affichage
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("RÉPONSE DE QWEN")
    print("=" * 60)
    print(answer)

    if TEXT_PATH.exists():
        print("\n" + "=" * 60)
        print("GROUND TRUTH")
        print("=" * 60)
        print(TEXT_PATH.read_text()[:2000])
        if TEXT_PATH.stat().st_size > 2000:
            print("\n... (tronqué)")

    print("\n✅ Test terminé")


if __name__ == "__main__":
    main()
