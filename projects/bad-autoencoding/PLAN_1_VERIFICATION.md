# Plan 1 — Reproduction Language Modeling (continuation perplexity)
## Objectif
Vérifier que le pipeline d'entraînement fonctionne et reproduire deux configurations clés du papier "Bad Autoencoding" (Table 2) :
1. Hierarchical (conv1d_residual) avec compression_target=63
2. Vision tiny frozen

Ces deux runs permettent de valider le setup avant de lancer les configurations plus lourdes.

**Environnement** : déjà prêt. Toutes les commandes s'exécutent depuis le répertoire `bad-autoencoding/`.

---

## Données
- Train : `./data/training/full/train.jsonl` (500 000 samples)
- Val : `./data/training/full/val.jsonl` (10 000 samples)

---

## Configurations à exécuter

| Nom | Régime | Flags principaux | Cible PPL (Table 2) | Commentaire |
|-----|--------|------------------|---------------------|-------------|
| Hierarchical t=63 | `conv1d_residual` | `--compression_target 63` | **5.14** | Baseline text-space (pas de vision) |
| Vision tiny frozen | `vision` | `--vision_mode tiny` | **5.35** | Vision encoder gelé, mode tiny (~12.8× compression) |

Aucune des deux configurations n'utilise `hybrid_text_tokens` (k=0, pure compression).

---

## Commandes à exécuter (dans l'ordre)

### 1. Hierarchical t=63

```bash
uv run python train.py \
  --regime conv1d_residual \
  --compression_target 63 \
  --objective lm \
  --num_epochs 1 \
  --learning_rate 1e-4 \
  --data_path ./data/training/full/train.jsonl \
  --val_data_path ./data/training/full/val.jsonl \
  --output_dir ./outputs/conv1d_t63_lm \
  --batch_size 8 \
  --gradient_accumulation_steps 6
```

### 2. Vision tiny frozen

```bash
uv run python train.py \
  --regime vision \
  --vision_mode tiny \
  --objective lm \
  --num_epochs 1 \
  --learning_rate 1e-4 \
  --data_path ./data/training/full/train.jsonl \
  --val_data_path ./data/training/full/val.jsonl \
  --output_dir ./outputs/vision_tiny_frozen_lm \
  --batch_size 8 \
  --gradient_accumulation_steps 6
```

**Notes** :
- `effective_batch_size` = 8 × 6 = 48
- 1 epoch sur 500k samples
- Les checkpoints et logs sont écrits dans le `--output_dir` correspondant

---

## Chiffres de référence (Table 2 du papier)

| Configuration | Tokens | Comp. | PPL | ΔPPL vs full |
|---------------|--------|-------|-----|--------------|
| Hierarchical t=63 | 65 | 15.4× | **5.14** | +0.34 |
| Vision tiny frozen | 78 | 12.8× | **5.35** | +0.55 |

---

## Format de log à produire et transmettre

À la fin de chaque run, extraire et formater les informations suivantes (un bloc par configuration) :

```
=== LM RUN ===
regime: conv1d_residual|vision
config: compression_target=63 | vision_mode=tiny
hybrid_text_tokens: 0
train_encoder: false
init: recon|DS
effective_batch: 48
tokens_seen_by_decoder: <nombre exact>
compression_ratio: <X.X>×

=== VAL FINAL ===
val_loss: <X.XXXX>
val_perplexity: <X.XX>
checkpoint: <chemin relatif vers le checkpoint>

=== VS PAPER ===
paper_config: hierarchical t=63 | vision tiny frozen
paper_ppl: 5.14 | 5.35
delta_ppl: <mon_ppl - paper_ppl>
```

**Comment obtenir les valeurs** :
- `val_loss` et `val_perplexity` : dans les logs finaux du script ou dans le fichier de métriques du output_dir
- `tokens_seen_by_decoder` et `compression_ratio` : généralement affichés dans les logs de démarrage ou de validation
- `checkpoint` : chemin du dernier checkpoint sauvegardé dans `output_dir`

Transmettre les deux blocs une fois les runs terminés.
