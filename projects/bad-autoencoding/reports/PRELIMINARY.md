# PLAN_1 — Rapport préliminaire

**Statut : runs complets en cours. Les PPL finaux ne sont pas encore disponibles.**

Date de démarrage : 2026-10-23 00:00 UTC
Dépôt cible : `thunderbyte-labs/optizip`, branche `main`, commit `f11d2e7`
Sous-dossier : `projects/bad-autoencoding/`

---

## 1. Objectif

Reproduire deux lignes de la Table 2 du papier *Optical Context Compression Is
Just (Bad) Autoencoding* :

| Run | Régime | Initialisation | Tokens de contexte | PPL papier |
|---|---|---|---:|---:|
| Hierarchical t=63 | `conv1d_residual` | checkpoint de reconstruction | 65 | **5,14** |
| Vision tiny frozen | `vision` | décodeur DeepSeek-OCR | 78 | **5,35** |

Les deux runs utilisent `objective=lm`, `hybrid_text_tokens=0`, un batch
effectif de 48 et un epoch sur 500 000 échantillons.

---

## 2. Environnement validé

| Élément | Valeur |
|---|---|
| GPU | NVIDIA GB10, 121,7 Go, compute capability 12.1 |
| CUDA | 13.0 |
| PyTorch | 2.13.0+cu130 |
| Transformers | 4.46.3 |
| Attention | `eager` (pas de wheel FlashAttention-2 aarch64 + CUDA 13) |
| Tests | 6 tests pytest passent |
| Données | `train_arrow` 500 000 ex. (10,18 Go), `val_arrow` 10 000 ex. (0,20 Go) |
| Poids | DeepSeek-OCR 6,3 Go + checkpoint recon t=63 18 Go |

### Smoke tests

| Test | Tokens de contexte | Ratio | Résultat |
|---|---:|---:|---|
| Vision tiny frozen | **78** (73 vision + 4 prompt + BOS) | 12,82× | ✅ |
| Conv1d t=63 | **65** (63 + séparateur + BOS) | 15,38× | ✅ |

Les deux smoke tests chargent les vraies données Arrow et les vraies images.
Le checkpoint de reconstruction t=63 se charge correctement, avec le switch
d'objectif `reconstruction → lm`.

---

## 3. Débit mesuré et durée estimée

Benchmark sur le GB10, attention eager :

| Batch | Débit observé | Temps par batch | Estimation par update (batch effectif 48) |
|---:|---:|---:|---:|
| 1 | ~1 000 tok/s | ~1,0 s | — |
| 8 | ~1 680 tok/s | ~4,8 s | ~29 s |
| 16 | ~1 740 tok/s | ~10 s | ~30 s |

Le batch 16 n'apporte pas de gain : le modèle est limité par le calcul, pas par
le débit de données.

Estimation : **10 417 updates × ~29 s ≈ 3,5 jours** par run, plus les
validations (21 passes de 1 250 batches) et la validation finale avec
log-probabilités par échantillon. Prévoir **~4 jours par run**, soit **~8 jours
pour les deux configurations**.

---

## 4. État des runs

| Run | Démarrage | Statut | Sortie |
|---|---|---|---|
| Conv1d t=63 | 2026-10-23 00:00 UTC | validation initiale puis entraînement | `outputs/conv1d_t63_lm_recon-init/` |
| Vision tiny frozen | après le run conv | en attente | `outputs/vision_tiny_frozen_lm/` |

Les runs sont lancés dans une session tmux détachée (`plan1`). Un second
processus (`plan1-report`) attend la fin des deux runs puis génère
automatiquement le rapport statistique complet dans `outputs/report/`.

Suivi :

```bash
tmux attach -t plan1
tail -f outputs/plan1_run_all.log
tail -f outputs/conv1d_t63_lm_recon-init/train.log
```

Les sorties sont aussi visibles dans `outputs/plan1_run_all.log`. Le run conv
tourne à ~95 % d'utilisation GPU.

---

## 5. Analyse statistique prévue

Chaque validation finale écrit `val_per_sample.jsonl` (une NLL moyenne par
échantillon). Le rapport final calculera :

1. **PPL et IC à 95 %** par bootstrap sur les 10 000 échantillons de validation,
   séparément pour chaque run.
2. **Comparaison au papier** : ΔPPL et position de la valeur du papier dans
   l'IC. C'est une comparaison descriptive : le papier ne publie qu'une
   estimation ponctuelle, sans données par échantillon.
3. **Comparaison appariée hierarchical vs vision** sur les mêmes échantillons :
   - ΔNLL et ΔPPL appariés, IC bootstrap ;
   - test de permutation par changement de signe (10 000 permutations) ;
   - fraction d'échantillons où le conv1d est meilleur ;
   - taille d'effet de Cohen $d_z$.
4. **Courbes de validation** et distribution des NLL par échantillon.

### Ce que ces tests peuvent et ne peuvent pas dire

- L'IC intra-run mesure l'incertitude d'échantillonnage de la validation, pas la
  variance entre graines d'entraînement. Un seul run par configuration ne permet
  pas d'estimer cette dernière.
- La comparaison au papier n'est pas un test à deux échantillons : le papier ne
  fournit pas les sorties par échantillon. Une valeur du papier hors de notre IC
  signifie que l'écart dépasse le bruit de validation, pas qu'il est
  exclusivement dû à la méthode.
- La comparaison appariée conv vs vision est, elle, un vrai test sur les mêmes
  échantillons. Un ΔPPL négatif signifie que l'encodeur hiérarchique est
  meilleur, ce qui est la revendication centrale du papier.

---

## 6. Rapport final

Le rapport final sera généré automatiquement à la fin des runs :

```text
outputs/report/REPORT.md
outputs/report/summary.json
outputs/report/fig1_ppl_vs_paper.png
outputs/report/fig2_delta_vs_paper.png
outputs/report/fig3_bootstrap_ppl.png
outputs/report/fig4_paired_delta.png
outputs/report/fig5_validation_curves.png
outputs/report/fig6_per_sample_ecdf.png
```

Pour le régénérer manuellement :

```bash
cd projects/bad-autoencoding
DATA_DIR=/home/jules/Documents/bad-autoencoding/data/training/full \
OUTPUT_ROOT=$PWD/outputs \
uv run python scripts/reproduce_plan1/make_report.py
```

---

## 7. Reproductibilité

```bash
cd projects/bad-autoencoding
bash scripts/reproduce_plan1/00_setup_gb10.sh
bash scripts/reproduce_plan1/01_smoke_vision.sh
bash scripts/reproduce_plan1/01_smoke_conv1d.sh
bash scripts/reproduce_plan1/run_all.sh
uv run python scripts/reproduce_plan1/make_report.py
```

Les commandes complètes, les flags et les variables d'environnement sont
documentés dans `scripts/reproduce_plan1/README.md`.
