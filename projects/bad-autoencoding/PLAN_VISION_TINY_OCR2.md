# Plan — Vision tiny frozen avec DeepSeek-OCR-2 (encoder + decoder)

## Objectif
Reproduire le run **vision tiny frozen** (continuation perplexity) du papier, mais en utilisant **l’encoder vision et le décodeur** de **DeepSeek-OCR-2** au lieu de DeepSeek-OCR.

Repo OCR-2 local : `/home/jules/Documents/DeepSeek-OCR-2`

Tu peux y créer une branche si des modifications sont nécessaires.

---

## Contexte technique

- Dans le papier, le régime `vision` + `vision_mode tiny` utilise l’encoder vision de DeepSeek-OCR (gelé) + son décodeur.
- On veut remplacer les deux par ceux de DeepSeek-OCR-2.
- DeepEncoder V2 (OCR-2) produit des tokens visuels via un mécanisme de causal flow (≈256 tokens selon la résolution).
- Le décodeur OCR-2 est un DeepSeek-V2 MoE (hidden size 1280).

Le mode "tiny" devra être mappé sur la résolution / le nombre de tokens le plus proche disponible dans OCR-2 (probablement la config par défaut ou la plus petite).

---

## Étapes pour OpenCode

### 1. Identifier le chargement du modèle vision

```bash
cd ~/Documents/bad-autoencoding
rg -n "DeepSeek-OCR|vision|from_pretrained|AutoModel" --type py
```

### 2. Créer une branche

```bash
git checkout -b vision-tiny-ocr2
```

### 3. Adapter le chargement

- Charger le modèle complet DeepSeek-OCR-2 (encoder + decoder) depuis le chemin local ou Hugging Face `deepseek-ai/DeepSeek-OCR-2`.
- Adapter le code du régime `vision` pour utiliser cet encoder (DeepEncoder V2) au lieu de celui d’OCR-1.
- S’assurer que le mode "tiny" correspond à une résolution / un nombre de tokens raisonnable dans OCR-2 (documenter le nombre de tokens produit).
- Le décodeur est gelé ou non selon le flag `--train_encoder` (ici on veut frozen, donc pas de `--train_encoder`).

### 4. Lancer le run

```bash
uv run python train.py \
  --regime vision \
  --vision_mode tiny \
  --objective lm \
  --num_epochs 1 \
  --learning_rate 1e-4 \
  --data_path ./data/training/full/train.jsonl \
  --val_data_path ./data/training/full/val.jsonl \
  --output_dir ./outputs/vision_tiny_frozen_ocr2_lm \
  --batch_size 8 --gradient_accumulation_steps 6
```

(Ajuster `--vision_mode` si OCR-2 utilise d’autres noms de modes.)

### 5. Format de log attendu

```
=== LM RUN (OCR-2) ===
regime: vision
config: vision_mode=tiny (OCR-2)
model: DeepSeek-OCR-2 (encoder + decoder)
train_encoder: false
hybrid_text_tokens: 0
effective_batch: 48
tokens_produced_by_encoder: <nombre>
compression_ratio: <X.X>×

=== VAL FINAL ===
val_loss: <X.XXXX>
val_perplexity: <X.XX>
checkpoint: ./outputs/vision_tiny_frozen_ocr2_lm/...

=== VS PAPER (OCR-1 vision tiny frozen) ===
paper_ppl: 5.35
delta_ppl: <mon_ppl - 5.35>
```

---

## Notes

- Le but est de mesurer si l’encoder DeepEncoder V2 (causal flow) change la PPL de continuation par rapport à l’encoder OCR-1.
- Si le mapping "tiny" n’existe pas dans OCR-2, utiliser la résolution qui produit le nombre de tokens le plus proche de 78 (ou documenter le nombre réel).
- Comparer ensuite avec le résultat OCR-1 déjà obtenu (5.45).
