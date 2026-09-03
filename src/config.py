"""Rutas y constantes compartidas por todos los scripts del laboratorio."""
from pathlib import Path

import nltk

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "outputs" / "figures"
TABLES_DIR = ROOT / "outputs" / "tables"
NETWORK_DIR = ROOT / "outputs" / "tables" / "network"
NLTK_DATA_DIR = ROOT / "nltk_data"

for d in (PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, NETWORK_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Se antepone la ruta local del proyecto para que nltk encuentre stopwords
# sin depender de la instalacion por defecto del sistema (ver setup_nltk.py).
if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))

VIDEOS_RAW_CSV = RAW_DIR / "youtube_videos.csv"
COMMENTS_RAW_CSV = RAW_DIR / "youtube_comments.csv"

VIDEOS_CLEAN_CSV = PROCESSED_DIR / "videos_clean.csv"
COMMENTS_CLEAN_CSV = PROCESSED_DIR / "comments_clean.csv"
MERGED_CSV = PROCESSED_DIR / "comments_videos_merged.csv"

RANDOM_SEED = 42
