"""Descarga los recursos de NLTK (stopwords) a nltk_data/ dentro del proyecto.

Se descarga a una carpeta local (en vez de la ruta por defecto del usuario)
porque en el equipo de desarrollo la ruta por defecto puede apuntar a una
unidad bloqueada (BitLocker) y para mantener el proyecto autocontenido.
"""
import nltk

from config import NLTK_DATA_DIR


def main():
    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    nltk.data.path = [str(NLTK_DATA_DIR)]
    nltk.download("stopwords", download_dir=str(NLTK_DATA_DIR))
    print(f"Recursos de NLTK descargados en: {NLTK_DATA_DIR}")


if __name__ == "__main__":
    main()
