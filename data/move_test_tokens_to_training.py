#!/usr/bin/env python3
"""
Déplace rapidement les triplets (code_xxxx.pt, code_xxxx.txt, code_xxxx.png)
de test/ vers training/ en se basant uniquement sur les .pt présents dans
test/tokens_896/.

À exécuter depuis le répertoire parent de test/ et training/ (ex: data/) :

    python move_test_tokens_to_training.py

Optimisé pour Linux : utilise os.rename (très rapide, même FS).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    base = Path(".").resolve()

    test_tokens = base / "test" / "tokens_896"
    test_texts  = base / "test" / "texts"
    test_images = base / "test" / "images"

    train_tokens = base / "training" / "tokens_896"
    train_texts  = base / "training" / "texts"
    train_images = base / "training" / "images"

    # Vérifications minimales
    if not test_tokens.is_dir():
        print(f"ERREUR: {test_tokens} n'existe pas", file=sys.stderr)
        return 1

    # Créer les dossiers de destination s'ils n'existent pas
    for d in (train_tokens, train_texts, train_images):
        d.mkdir(parents=True, exist_ok=True)

    # Lister tous les code_*.pt (scandir est le plus rapide)
    pt_files = [
        entry.name
        for entry in os.scandir(test_tokens)
        if entry.is_file() and entry.name.startswith("code_") and entry.name.endswith(".pt")
    ]

    n = len(pt_files)
    print(f"Trouvé {n} fichiers .pt dans {test_tokens}")

    if n == 0:
        return 0

    moved = 0
    missing_txt = 0
    missing_png = 0
    errors = 0

    for i, pt_name in enumerate(pt_files, 1):
        stem = pt_name[:-3]  # enlever ".pt" → code_xxxx

        src_pt  = test_tokens / pt_name
        src_txt = test_texts  / f"{stem}.txt"
        src_png = test_images / f"{stem}.png"

        dst_pt  = train_tokens / pt_name
        dst_txt = train_texts  / f"{stem}.txt"
        dst_png = train_images / f"{stem}.png"

        try:
            # .pt (toujours présent car on l'a listé)
            os.rename(src_pt, dst_pt)

            # .txt
            if src_txt.is_file():
                os.rename(src_txt, dst_txt)
            else:
                missing_txt += 1

            # .png
            if src_png.is_file():
                os.rename(src_png, dst_png)
            else:
                missing_png += 1

            moved += 1
        except OSError as e:
            print(f"  [ERREUR] {stem}: {e}", file=sys.stderr)
            errors += 1

        # progression légère
        if i % 500 == 0 or i == n:
            print(f"  … {i}/{n}")

    print()
    print(f"Terminé.")
    print(f"  Triplets déplacés : {moved}")
    if missing_txt:
        print(f"  .txt manquants   : {missing_txt}")
    if missing_png:
        print(f"  .png manquants   : {missing_png}")
    if errors:
        print(f"  Erreurs          : {errors}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

