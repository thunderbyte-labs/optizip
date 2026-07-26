#!/usr/bin/env python3
"""
Génération de paires image/texte pour l'entraînement de l'OpticalAdapter
à partir de MULTIPLES codebases (7 langages).

Stratégies :
1. Capture de code avec Pygments (syntax highlighting réaliste)
2. Découpage intelligent : ne pas couper les fonctions/classes en plein milieu
3. Génération de "mini-documents" : plusieurs fonctions dans une même page
4. Variation de résolution et de taille de police (robustesse)
5. Support multi-langages : C++, Python, Rust, C, ASM (x86), Java, C#
6. Sampling massif : jusqu'à 100k+ paires
"""

import os
import sys
import random
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import defaultdict

from PIL import Image
from pygments import highlight
from pygments.lexers import (
    get_lexer_for_filename,
    CppLexer, PythonLexer, RustLexer, CLexer,
    NasmLexer, JavaLexer, CSharpLexer,
)
from pygments.formatters import ImageFormatter
from pygments.styles import get_style_by_name
from tqdm import tqdm

# ===================== Configuration =====================
@dataclass
class DatasetEntry:
    """Définition d'un dataset source."""
    name: str
    source_root: str
    include_dirs: List[str]
    extensions: List[str]
    default_lexer: type
    comment_syntax: str = "//"  # pour détecter les commentaires
    weight: float = 1.0  # pondération pour l'échantillonnage

@dataclass
class Config:
    # Sortie
    output_dir: str = "data/optical_alignment_code"
    images_dir: str = "data/optical_alignment_code/images"
    texts_dir: str = "data/optical_alignment_code/texts"
    
    # Datasets sources
    datasets: List[DatasetEntry] = field(default_factory=lambda: [
        # C++ - Firefox/Gecko (le plus riche en métaprogrammation)
        DatasetEntry(
            name="cpp_firefox",
            source_root="/home/jules/Documents/code-image-text-gen/firefox-meta",
            include_dirs=["mfbt", "dom/bindings", "xpcom/string", "layout/style"],
            extensions=[".h", ".cpp", ".hpp", ".cc", ".cxx"],
            default_lexer=CppLexer,
            comment_syntax="//",
            weight=2.5,
        ),
        # Python - CPython
        DatasetEntry(
            name="python_cpython",
            source_root="/home/jules/Documents/code-image-text-gen/cpython-meta",
            include_dirs=["Lib", "Objects", "Python"],
            extensions=[".py", ".pyi"],
            default_lexer=PythonLexer,
            comment_syntax="#",
            weight=1.5,
        ),
        # Rust - Compilateur Rust
        DatasetEntry(
            name="rust_rustc",
            source_root="/home/jules/Documents/code-image-text-gen/rust-meta",
            include_dirs=["library/core/src", "library/std/src", "library/alloc/src",
                         "compiler/rustc_macros", "compiler/rustc_builtin_macros"],
            extensions=[".rs"],
            default_lexer=RustLexer,
            comment_syntax="//",
            weight=1.5,
        ),
        # C - Linux Kernel
        DatasetEntry(
            name="c_linux",
            source_root="/home/jules/Documents/code-image-text-gen/linux-meta",
            include_dirs=["include/linux", "include/asm-generic", "kernel/locking", "lib"],
            extensions=[".h", ".c"],
            default_lexer=CLexer,
            comment_syntax="//",
            weight=1.5,
        ),
        # ASM - NASM
        DatasetEntry(
            name="asm_nasm",
            source_root="/home/jules/Documents/code-image-text-gen/nasm-meta",
            include_dirs=["asm", "macros"],
            extensions=[".asm", ".nasm", ".mac", ".inc"],
            default_lexer=NasmLexer,
            comment_syntax=";",
            weight=0.8,
        ),
        # Java - Spring Framework
        DatasetEntry(
            name="java_spring",
            source_root="/home/jules/Documents/code-image-text-gen/spring-meta",
            include_dirs=[
                "spring-core/src/main/java",
                "spring-aop/src/main/java",
                "spring-expression/src/main/java",
            ],
            extensions=[".java"],
            default_lexer=JavaLexer,
            comment_syntax="//",
            weight=1.2,
        ),
        # C# - Roslyn
        DatasetEntry(
            name="csharp_roslyn",
            source_root="/home/jules/Documents/code-image-text-gen/roslyn-meta",
            include_dirs=[
                "src/Compilers/CSharp/Portable",
                "src/Features/Core/Portable",
                "src/Workspaces/Core/Portable",
                "src/Tools/Source",
            ],
            extensions=[".cs"],
            default_lexer=CSharpLexer,
            comment_syntax="//",
            weight=1.0,
        ),
    ])
    
    # Paramètres de génération
    min_lines: int = 15
    max_lines: int = 250
    max_page_chars: int = 15000
    
    # Résolutions (variation pour robustesse)
    font_sizes: List[int] = field(default_factory=lambda: [10, 11, 12, 13, 14, 15, 16, 18, 20])
    dpi_variations: List[int] = field(default_factory=lambda: [120, 150, 200, 250, 300])
    
    # Styles Pygments (diversité visuelle)
    styles: List[str] = field(default_factory=lambda: [
        "monokai", "dracula", "one-dark", "native", "vs", "xcode",
        "friendly", "autumn", "borland", "tango", "emacs", "vim",
        "murphy", "pastie", "perldoc", "rainbow_dash", "solarized-dark",
        "solarized-light", "zenburn",
    ])
    
    # Nombre total de paires à générer
    num_samples: int = 50000
    
    # Paramètres d'image
    font_name: str = "DejaVu Sans Mono"
    line_numbers: bool = True
    line_pad: int = 4
    
    # Multiprocessing
    num_workers: int = max(1, os.cpu_count() - 1)
    
    # Seed pour reproductibilité
    seed: int = 42

# ===================== Parsing de code multi-langages =====================
class MultiLangCodeParser:
    """Découpe intelligemment le code source en blocs cohérents, multi-langages."""
    
    # Patterns par langage
    FUNCTION_PATTERNS = {
        "default": re.compile(
            r'^\s*(?:'
            r'(?:virtual\s+|static\s+|inline\s+|constexpr\s+|explicit\s+|'
            r'pub\s+|pub\([^)]*\)\s+|unsafe\s+|extern\s+|'
            r'async\s+|fn\s+|def\s+|public\s+|private\s+|protected\s+|'
            r'internal\s+|static\s+|final\s+|abstract\s+|override\s+|'
            r'virtual\s+|synchronized\s+|native\s+|strictfp\s+)*'
            r'(?:[\w:<>&*,\s\[\]]+\s+)?'           # type de retour
            r'(?:[\w:]+::)?'                        # namespace/classe
            r'~?[\w]+\s*\([^)]*\)\s*'               # nom fonction(params)
            r'(?:const\s*)?'                        # const
            r'(?:override\s*|final\s*|noexcept\s*)*'
            r'(?:;|\{|->)'
            r')',
            re.MULTILINE
        ),
        "python": re.compile(
            r'^\s*(?:'
            r'(?:async\s+)?def\s+\w+\s*\(|'         # def fonction(
            r'class\s+\w+|'                          # class Nom
            r'@\w+'                                   # décorateur
            r')',
            re.MULTILINE
        ),
        "asm": re.compile(
            r'^\s*(?:'
            r'(?:global|extern|section|bits|default|'
            r'%macro|%define|%if|%else|%elif|%endif|'
            r'[a-zA-Z_]\w*:)'                         # label:
            r')',
            re.MULTILINE | re.IGNORECASE
        ),
    }
    
    CLASS_PATTERNS = {
        "default": re.compile(
            r'^\s*(?:class|struct|namespace|enum\s+class|enum|interface|'
            r'trait|impl|object|module|%macro)\s+\w+',
            re.MULTILINE
        ),
        "python": re.compile(r'^\s*class\s+\w+', re.MULTILINE),
        "asm": re.compile(r'^\s*%macro\s+\w+', re.MULTILINE | re.IGNORECASE),
    }
    
    @classmethod
    def get_function_pattern(cls, lang_name: str) -> re.Pattern:
        if lang_name.startswith("python"):
            return cls.FUNCTION_PATTERNS["python"]
        elif lang_name.startswith("asm"):
            return cls.FUNCTION_PATTERNS["asm"]
        return cls.FUNCTION_PATTERNS["default"]
    
    @classmethod
    def get_class_pattern(cls, lang_name: str) -> re.Pattern:
        if lang_name.startswith("python"):
            return cls.CLASS_PATTERNS["python"]
        elif lang_name.startswith("asm"):
            return cls.CLASS_PATTERNS["asm"]
        return cls.CLASS_PATTERNS["default"]
    
    @staticmethod
    def find_smart_split(lines: List[str], target_lines: int,
                         comment_char: str = "//") -> int:
        """Trouve le meilleur endroit pour couper."""
        if target_lines >= len(lines):
            return len(lines)
        
        search_range = min(30, target_lines // 3)
        best_split = target_lines
        best_score = float('inf')
        
        for offset in range(-search_range, search_range + 1):
            idx = target_lines + offset
            if idx <= 0 or idx >= len(lines):
                continue
            
            line = lines[idx].strip()
            score = 0
            
            # Favoriser les lignes vides
            if line == '':
                score -= 15
            
            # Favoriser les fins de bloc (ligne avec juste })
            if line in ('}', ');', 'end', 'endif', '%endmacro', '%endif'):
                score -= 12
            
            # Éviter de couper en plein commentaire
            if line.startswith((comment_char, '/*', '*', '#')):
                score += 25
            
            # Éviter de couper dans une chaîne ouverte
            if line.count('"') % 2 != 0 or line.count("'") % 2 != 0:
                score += 20
            
            # Favoriser les débuts de fonction/classe
            if re.match(r'^\s*(def |class |fn |pub |%macro|\w+:)', line):
                score -= 8
            
            if score < best_score:
                best_score = score
                best_split = idx
        
        return best_split
    
    @staticmethod
    def extract_snippets(filepath: Path, min_lines: int, max_lines: int,
                         lang_name: str, comment_char: str) -> List[str]:
        """Extrait des snippets de code d'un fichier."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return []
        
        if len(content) < min_lines * 8:
            return []
        
        lines = content.split('\n')
        func_pattern = MultiLangCodeParser.get_function_pattern(lang_name)
        class_pattern = MultiLangCodeParser.get_class_pattern(lang_name)
        
        # Trouver des points de départ naturels
        start_candidates = []
        for i, line in enumerate(lines):
            if func_pattern.match(line) or class_pattern.match(line):
                start_candidates.append(i)
        
        # Si pas assez de candidats, ajouter des points aléatoires
        if len(start_candidates) < 10 and len(lines) > min_lines:
            population_size = len(lines) - min_lines
            if population_size > 0:
                sample_size = min(50, max(10, len(lines) // 30))
                if sample_size > population_size:
                    sample_size = population_size
                extra = random.sample(range(population_size), sample_size)
                start_candidates.extend(extra)
        
        if not start_candidates:
            return []
        
        # Mélanger et limiter
        random.shuffle(start_candidates)
        start_candidates = start_candidates[:100]
        
        snippets = []
        for start in start_candidates:
            target_end = start + random.randint(min_lines, max_lines)
            end = MultiLangCodeParser.find_smart_split(
                lines, target_end, comment_char
            )
            
            snippet = '\n'.join(lines[start:end])
            snippet_lines = snippet.split('\n')
            
            if min_lines <= len(snippet_lines) <= max_lines:
                # Vérifier que le snippet contient du code substantiel
                code_lines = [
                    l for l in snippet_lines
                    if l.strip() and not l.strip().startswith((comment_char, '#', ';', '/*', '*'))
                ]
                if len(code_lines) >= min_lines * 0.4:
                    snippets.append(snippet)
        
        return snippets

# ===================== Génération d'images =====================
class CodeImageGenerator:
    """Génère des images de code avec variation de style."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def render_code_to_image(
        self, 
        code: str,
        style_name: str,
        font_size: int,
        dpi: int,
        lexer,                # <-- peut être une classe ou une instance
        line_numbers: bool = True,
    ) -> Optional[Image.Image]:
        """Rend le code en image avec Pygments."""
        try:
            # ✅ INSTANCIE LE LEXER SI C'EST UNE CLASSE
            if isinstance(lexer, type):
                lexer = lexer()
            
            style = get_style_by_name(style_name)
            
            # Certains styles n'ont pas de background_color
            bg_color = getattr(style, 'background_color', None) or '#1e1e1e'
            
            formatter = ImageFormatter(
                font_name=self.config.font_name,
                font_size=font_size,
                line_numbers=line_numbers,
                line_pad=self.config.line_pad,
                style=style,
                image_format="png",
                dpi=dpi,
                line_number_bg=bg_color,
                line_number_fg='#888888',
            )
            
            image_data = highlight(code, lexer, formatter)
            
            from io import BytesIO
            image = Image.open(BytesIO(image_data))
            return image
            
        except Exception as e:
            import traceback
            print(f"\n[RENDER ERROR] style={style_name}, font_size={font_size}, dpi={dpi}")
            traceback.print_exc()
            return None

    def save_pair(self, image: Image.Image, code: str, index: int):
        """Sauvegarde une paire image/texte."""
        os.makedirs(self.config.images_dir, exist_ok=True)
        os.makedirs(self.config.texts_dir, exist_ok=True)
        
        img_path = Path(self.config.images_dir) / f"code_{index:08d}.png"
        txt_path = Path(self.config.texts_dir) / f"code_{index:08d}.txt"
        
        image.save(img_path, "PNG")
        txt_path.write_text(code, encoding='utf-8')

# ===================== Worker multiprocessing =====================
def worker_process(args):
    """Fonction exécutée par chaque worker."""
    dataset_entry, config_dict, worker_id, num_workers = args
    
    # Reconstruire config
    config = Config()
    for k, v in config_dict.items():
        if k != 'datasets':
            setattr(config, k, v)
    
    dataset = DatasetEntry(**dataset_entry)
    parser = MultiLangCodeParser()
    
    # Collecter les fichiers
    all_files = []
    for d in dataset.include_dirs:
        dir_path = Path(dataset.source_root) / d
        if not dir_path.exists():
            continue
        for ext in dataset.extensions:
            all_files.extend(dir_path.rglob(f"*{ext}"))
    
    # Filtrer selon worker_id
    random.seed(config.seed + worker_id)
    random.shuffle(all_files)
    all_files = all_files[worker_id::num_workers]
    
    # Extraire les snippets
    snippets = []
    for filepath in all_files:
        file_snippets = parser.extract_snippets(
            filepath,
            config.min_lines,
            config.max_lines,
            dataset.name,
            dataset.comment_syntax,
        )
        snippets.extend(file_snippets)
    
    return snippets, dataset.name, dataset.default_lexer

# ===================== Main =====================
import multiprocessing as mp

def generate_dataset(config: Config):
    """Fonction principale de génération multi-dataset."""
    
    print("=" * 70)
    print("🎯 GÉNÉRATION DU DATASET CODE MULTI-LANGAGES POUR OPTICALADAPTER")
    print("=" * 70)
    print(f"Cible: {config.num_samples:,} paires image/texte")
    print(f"Langages: {len(config.datasets)}")
    print(f"Workers: {config.num_workers}")
    print()
    
    # Afficher les datasets
    total_weight = sum(d.weight for d in config.datasets)
    print("📦 Datasets disponibles:")
    for ds in config.datasets:
        allocated = int(config.num_samples * ds.weight / total_weight)
        print(f"   {ds.name:20s} → {allocated:>8,} paires (poids: {ds.weight})")
    print()
    
    # Phase 1 : Extraction parallèle des snippets
    print("📝 Phase 1: Extraction parallèle des snippets...")
    
    config_dict = {
        k: v for k, v in config.__dict__.items()
        if k != 'datasets'
    }
    
    all_snippets_by_dataset: Dict[str, List[str]] = defaultdict(list)
    dataset_lexers: Dict[str, type] = {}
    
    # Préparer les tâches
    tasks = []
    for ds in config.datasets:
        ds_dict = {
            'name': ds.name,
            'source_root': ds.source_root,
            'include_dirs': ds.include_dirs,
            'extensions': ds.extensions,
            'default_lexer': ds.default_lexer.__name__,
            'comment_syntax': ds.comment_syntax,
            'weight': ds.weight,
        }
        dataset_lexers[ds.name] = ds.default_lexer
        for w in range(config.num_workers):
            tasks.append((ds_dict, config_dict, w, config.num_workers))
    
    # Exécuter en parallèle
    with mp.Pool(config.num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(worker_process, tasks),
            total=len(tasks),
            desc="   Extraction"
        ))
    
    for snippets, ds_name, _ in results:
        all_snippets_by_dataset[ds_name].extend(snippets)
    
    # Statistiques d'extraction
    print("\n📊 Snippets extraits par dataset:")
    for ds_name, snippets in all_snippets_by_dataset.items():
        print(f"   {ds_name:20s}: {len(snippets):>8,} snippets")
    
    total_extracted = sum(len(s) for s in all_snippets_by_dataset.values())
    print(f"   {'TOTAL':20s}: {total_extracted:>8,} snippets\n")
    
    # Phase 2 : Échantillonnage proportionnel
    print("🎯 Phase 2: Échantillonnage proportionnel...")
    
    random.seed(config.seed)
    selected_snippets = []
    
    for ds in config.datasets:
        snippets = all_snippets_by_dataset[ds.name]
        n_allocated = max(1, int(config.num_samples * ds.weight / total_weight))
        n_available = len(snippets)
        
        if n_available == 0:
            print(f"   ⚠ {ds.name}: aucun snippet disponible")
            continue
        
        if n_allocated > n_available:
            # Suréchantillonner avec répétition
            sampled = random.choices(snippets, k=n_allocated)
            print(f"   {ds.name:20s}: {n_allocated:>6,} sélectionnés (suréchantillonnage, {n_available} disponibles)")
        else:
            sampled = random.sample(snippets, n_allocated)
            print(f"   {ds.name:20s}: {n_allocated:>6,} sélectionnés (sur {n_available} disponibles)")
        
        for s in sampled:
            selected_snippets.append((ds.name, s, dataset_lexers[ds.name]))
    
    # Mélanger tous les snippets
    random.shuffle(selected_snippets)
    
    # Limiter au total demandé
    if len(selected_snippets) > config.num_samples:
        selected_snippets = selected_snippets[:config.num_samples]
    
    print(f"   Total sélectionné: {len(selected_snippets):,} snippets\n")
    
    # Phase 3 : Génération des images
    print("🎨 Phase 3: Génération des images...")
    generator = CodeImageGenerator(config)
    generated = 0
    
    # Créer les dossiers
    os.makedirs(config.images_dir, exist_ok=True)
    os.makedirs(config.texts_dir, exist_ok=True)
    
    # Fichier de mapping pour traçabilité
    mapping_file = Path(config.output_dir) / "dataset_mapping.csv"
    with open(mapping_file, 'w') as f:
        f.write("index,dataset,style,font_size,dpi,line_numbers,char_count,line_count\n")
        
        for i, (ds_name, snippet, lexer) in enumerate(tqdm(selected_snippets, desc="   Rendu")):
            # Varier les paramètres
            style = random.choice(config.styles)
            font_size = random.choice(config.font_sizes)
            dpi = random.choice(config.dpi_variations)
            line_numbers = random.choice([True, True, False])  # biais vers True
            
            # Générer l'image
            image = generator.render_code_to_image(
                snippet, style, font_size, dpi, lexer, line_numbers
            )
            
            if image is None:
                continue
            
            # Sauvegarder
            generator.save_pair(image, snippet, generated)
            
            # Enregistrer le mapping
            snippet_lines = snippet.split('\n')
            f.write(f"{generated},{ds_name},{style},{font_size},{dpi},{line_numbers},"
                   f"{len(snippet)},{len(snippet_lines)}\n")
            
            generated += 1
    
    print(f"\n✅ Dataset généré: {generated:,} paires")
    print(f"   Images: {config.images_dir}/")
    print(f"   Textes: {config.texts_dir}/")
    print(f"   Mapping: {mapping_file}")
    
    # Statistiques finales
    print("\n📊 Statistiques finales:")
    img_files = list(Path(config.images_dir).glob("*.png"))
    txt_files = list(Path(config.texts_dir).glob("*.txt"))
    
    total_img_size = sum(f.stat().st_size for f in img_files)
    total_txt_size = sum(f.stat().st_size for f in txt_files)
    
    print(f"   Paires générées:     {generated:,}")
    print(f"   Taux de succès:      {generated / len(selected_snippets) * 100:.1f}%")
    print(f"   Taille images:       {total_img_size / 1024 / 1024:.1f} MB")
    print(f"   Taille textes:       {total_txt_size / 1024 / 1024:.1f} MB")
    print(f"   Taille moyenne img:  {total_img_size / generated / 1024:.1f} KB" if generated else "")
    print(f"   Taille moyenne txt:  {total_txt_size / generated / 1024:.1f} KB" if generated else "")
    
    # Distribution par dataset
    print("\n📈 Distribution par langage:")
    dist = defaultdict(int)
    with open(mapping_file) as f:
        next(f)  # skip header
        for line in f:
            ds = line.split(',')[1]
            dist[ds] += 1
    for ds, count in sorted(dist.items(), key=lambda x: -x[1]):
        pct = count / generated * 100 if generated else 0
        print(f"   {ds:20s}: {count:>8,} ({pct:5.1f}%)")
    
    # Résolution exemple
    if img_files:
        sample_img = Image.open(img_files[0])
        print(f"\n   Résolution exemple: {sample_img.size[0]}x{sample_img.size[1]} px")
    
    return generated


if __name__ == "__main__":
    config = Config()
    
    # Override depuis la ligne de commande
    if len(sys.argv) > 1:
        config.num_samples = int(sys.argv[1])
    if len(sys.argv) > 2:
        config.num_workers = int(sys.argv[2])
    
    # Créer les dossiers
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Générer
    num_generated = generate_dataset(config)
    
    if num_generated == 0:
        print("\n❌ Aucune paire générée. Vérifiez les chemins et les dépendances.")
        print("   pip install Pillow pygments tqdm")
        sys.exit(1)
    
    print(f"\n🚀 Dataset multi-langages prêt pour l'entraînement !")
    print(f"   python train_optical_adapter_phase1.py")
