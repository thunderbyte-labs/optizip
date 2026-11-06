# Rapport — Vision tiny frozen avec DeepSeek-OCR-2 (512, 14,3×)

**Statut** : ✅ run terminé le **6 novembre 2026 à 01:22 UTC**.
**Branche** : `vision-tiny-ocr2` (commit `079ff8c`, mergée sur `main`).
**Sortie** : `outputs/vision_tiny_frozen_ocr2_lm/`.
**Objectif** : reproduire la ligne « vision tiny frozen » du papier *Bad Autoencoding*
mais avec l'encodeur + décodeur de **DeepSeek-OCR-2** (DeepEncoder V2).

---

## TL;DR

| Métrique | Valeur |
|---|---|
| **PPL finale OCR-2 512 (14,3×)** | **5,94** — IC 99 % [5,81 ; 6,08] |
| PPL OCR-1 tiny (reproduit, 12,8×) | 5,45 — IC 99 % [5,33 ; 5,57] |
| PPL papier (OCR-1 tiny) | 5,35 |
| **Δ vs papier** | **+0,59** |
| **Δ vs OCR-1** | **+0,49** |
| Différence appariée OCR-2 − OCR-1 | **+0,0864 nat** → ratio PPL **1,090×** — IC 99 % [1,0886 ; 1,0920] |

> **Conclusion** : à ce budget (65 tokens visuels, 14,3×), OCR-2 est **statistiquement
> ~9 % moins bon** qu'OCR-1 tiny. Ce n'est pas du bruit : l'IC 99 % de la différence
> appariée exclut largement 1,0×. **Mais ce n'est pas un test propre de l'encodeur**
> (cf. §4) : budget plus agressif + `query_512` interpolé/gelé.

---

## 1. Configuration du run

| Paramètre | Valeur |
|---|---|
| Régime / objectif | `vision` / `lm` (continuation) |
| Modèle | DeepSeek-OCR-2 (DeepEncoder V2 + décodeur DeepSeek-V2) |
| Encodeur | **gelé** (`sam_model` + `qwen2_model` + `projector` + `view_seperator`) |
| Décodeur | fine-tuné (12 couches MoE, hidden 1280, 64+2 experts) |
| Résolution | 512×512 (`vision_mode=tiny`, `ocr_version=ocr2`) |
| Tokens visuels | **64 queries (`query_512`) + 1 `view_seperator` = 65** |
| Contexte effectif | 1 BOS + 65 vision + 4 prompt (`"Free OCR."`) = **70** |
| Compression | **14,29×** (10,0× côté vision seule) |
| Données | `train_arrow` (500 000) / `val_arrow` (10 000) |
| Batch | 8 × accumulation 6 = **48 effectif** |
| Steps | 10 417 |
| LR / schedule | 1e-4, warmup 1041, cosine |
| Attention | `eager` (GB10, pas de FlashAttention-2) |
| Seed | 42 |
| Durée | **84,7 h (3,53 jours)**, 1 639 tok/s |
| Matériel | NVIDIA GB10 (CUDA 13) |

**Paramètres** : total 3,389,176,704 (3,39 B) ; encodeur gelé 454,442,624 (13,4 %) ;
décodeur/embeddings entraînés 2,934,734,080 (86,6 %).

Initialisation `query_512` : interpolation **bicubique** de `query_1024`
(16×16 → 8×8), puis gelée (OCR-2 n'a pas de grille native 8×8).

---

## 2. Résultats finaux

- **Validation finale** (step 10 417) : `loss = 1,7816`, **`PPL = 5,94`**.
- Meilleur checkpoint : step 10 000, `val_loss = 1,7816` (stable sur les 500 derniers steps).
- Per-sample NLL (`val_per_sample.jsonl`, n = 10 000) :
  - moyenne = **1,781602 nat**, std = 0,8844, SE = 0,00884 ;
  - **IC 99 % (échantillonnage)** : PPL ∈ **[5,806 ; 6,076]**.
- Trajectoire : 10,07 (init) → 7,55 (500) → … → 5,94 (plateau dès ~step 8 000).

La courbe est monotone et converge proprement (pas de divergence MoE).

---

## 3. Comparaison à avancement égal vs OCR-1 tiny

Même protocole, mêmes données, même seed ; seule la config vision change.

| Global step | OCR-2 (14,3×) | OCR-1 (12,8×) | ΔPPL |
|---:|---:|---:|---:|
| 0 (initial) | 10,07 | 7,81 | +2,26 |
| 500 | 7,55 | 6,29 | +1,26 |
| 1000 | 7,22 | 6,53 | +0,69 |
| 1500 | 6,92 | 6,44 | +0,48 |
| 2000 | 6,71 | 6,29 | +0,42 |
| 2500 | 6,55 | 6,15 | +0,40 |
| 3000 | 6,42 | 6,03 | +0,39 |
| 3500 | 6,33 | 5,92 | +0,41 |
| 4000 | 6,24 | 5,83 | +0,41 |
| 4500 | 6,16 | 5,73 | +0,43 |
| 5000 | 6,10 | 5,66 | +0,44 |
| 5500 | 6,05 | 5,60 | +0,45 |
| 6000 | 6,01 | 5,55 | +0,46 |
| 6500 | 5,98 | 5,51 | +0,47 |
| 7000 | 5,96 | 5,48 | +0,48 |
| 7500 | 5,95 | 5,47 | +0,48 |
| 8000 | 5,94 | 5,46 | +0,48 |
| 8500 | 5,94 | 5,45 | +0,49 |
| 9000 | 5,94 | 5,45 | +0,49 |
| 9500 | 5,94 | 5,45 | +0,49 |
| 10000 | 5,94 | 5,45 | +0,49 |
| 10417 (final) | **5,94** | **5,45** | **+0,49** |

L'écart se referme fortement pendant le warmup (2,26 → 0,48), puis **se stabilise à
+0,49 PPL** (~+0,086 nat). OCR-2 ne rattrape donc pas OCR-1 ; il converge vers une
valeur systématiquement ~9 % plus haute.

### Test apparié (mêmes 10 000 échantillons de validation)

- Différence moyenne par échantillon : **+0,086439 nat** (std 0,0614, SE 0,000614).
- **IC 99 %** (normal + bootstrap, identiques) : **[0,0849 ; 0,0880] nat**.
- **Ratio PPL** : **1,0903×**, IC 99 % **[1,0886 ; 1,0920]**.
- Les IC non appariés eux-mêmes ne se recouvrent pas ([5,81 ; 6,08] vs [5,33 ; 5,57]).

---

## 4. Interprétation — pourquoi OCR-2 ne gagne pas ici

Ce run **n'est pas** un test propre de « DeepEncoder V2 est-il meilleur ? ». Trois
confonds, par ordre d'impact probable :

1. **Encodeur hors distribution (le plus gros)**. OCR-2 n'a été pré-entraîné qu'avec
   des grilles **12×12 (768 px)** et **16×16 (1024 px)**. Le `query_512` (8×8) est une
   interpolation bicubique **gelée** : les queries du *causal flow* encodent l'ordre de
   lecture et ne se factorisent pas spatialement. L'encodeur opère donc avec une grille
   qu'il n'a jamais vue → features dégradées.
2. **Budget de tokens plus agressif**. 65 tokens vs 73 pour OCR-1 (70 vs 78 avec
   BOS+prompt) → **14,3× vs 12,8×**. Une partie du ΔPPL est mécanique.
3. **Adéquation à la tâche**. Le *causal flow* vise l'OCR/document (layout, ordre de
   lecture), pas la continuation de prose Wikipedia. Rien ne garantit un gain ici.

À noter : le décodeur OCR-2 est un checkpoint **différent** d'OCR-1 (poids distincts
vérifiés, maxdiff 0,05–0,95), mais il est fine-tuné, donc ce facteur s'estompe.

**Ce que ce run établit** : à 14,3× de compression, l'OCR-2 « tiny hacké » plafonne à
5,94, soit ~9 % au-dessus d'OCR-1 tiny. **Ce qu'il n'établit pas** : la qualité de
l'encodeur OCR-2 en résolution native.

---

## 5. Artefacts

```
outputs/vision_tiny_frozen_ocr2_lm/
├── args.json                  # config complète (ocr_version=ocr2, vision_mode=tiny)
├── train.log                  # 158 Ko, 22 validations + logs
├── checkpoint_log.txt         # historique best/final
├── val_per_sample.jsonl       # 10 000 NLL par échantillon (657 Ko)
├── best_checkpoint.pt         # step 10000, val_loss 1,7816 (18,5 Go)
└── final_checkpoint.pt        # step 10417, val_loss 1,7816 (18,5 Go)
```

Référence OCR-1 : `outputs/vision_tiny_frozen_lm/` (PPL 5,45).

---

## 6. Prochaines étapes

Le test propre de l'encodeur est le plan **`OCR-2-768-6.7c.md`** : résolution
**native 768×768** (`query_768`, 12×12), **145 tokens visuels**, **6,67×** de
compression, images re-renderisées (font ~15 px), mode `ocr2_768`. Il lève le confond
« query interpolé » et teste DeepEncoder V2 dans sa distribution native — au prix d'un
budget non comparable au « tiny » du papier.

Alternative moins coûteuse : OCR-2 **1024 natif** (257 tokens, 3,8×) avec les images
`base` existantes, à comparer à OCR-1 base (273 tokens, 3,6×).

---

## 7. Reproductibilité

```bash
DATA_DIR=/shared/optizip-shared/bad-autoencoding/data/training/full \
  ./scripts/reproduce_plan1/07_vision_tiny_ocr2.sh
```

Le script appelle `train.py` avec `--ocr_version ocr2 --vision_mode tiny
--vision_prompt free_ocr --no-train-encoder`, batch 8 × accum 6, 1 epoch, LR 1e-4,
seed 42, `--attn_implementation eager`.

*Rapport généré le 2026-11-06 à partir de `train.log` et `val_per_sample.jsonl`.*
