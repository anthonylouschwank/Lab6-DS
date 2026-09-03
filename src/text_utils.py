"""Funciones de limpieza de texto reutilizadas por los scripts del laboratorio."""
import re
import ast

import emoji
from unidecode import unidecode
from nltk.corpus import stopwords

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[\w.\-]+")
HASHTAG_RE = re.compile(r"#\w+")
NON_ALPHA_RE = re.compile(r"[^a-zA-Záéíóúñü\s]")
MULTISPACE_RE = re.compile(r"\s+")

_SPANISH_STOPWORDS = set(stopwords.words("spanish"))
# Palabras muy frecuentes en comentarios de YouTube que no aportan a temas/sentimiento
_EXTRA_STOPWORDS = {"si", "q", "xq", "porq", "youtube", "video", "canal"}
STOPWORDS_ES = _SPANISH_STOPWORDS | _EXTRA_STOPWORDS


def parse_list_field(value: str) -> list:
    """Convierte campos tipo '["a", "b"]' (query_hits, keywords) a listas reales."""
    if value is None:
        return []
    value = str(value).strip()
    if value == "" or value == "[]":
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def extract_mentions(text: str) -> list:
    return MENTION_RE.findall(str(text))


def extract_hashtags(text: str) -> list:
    return HASHTAG_RE.findall(str(text))


def extract_emojis(text: str) -> list:
    return [c["emoji"] for c in emoji.emoji_list(str(text))]


def clean_text(text: str) -> str:
    """Pipeline de limpieza para texto_limpio (ver seccion 2.6 del informe).

    Orden documentado:
    1. minusculas
    2. eliminar URLs
    3. eliminar menciones (@usuario) y hashtags (#tema) ya extraidos aparte
    4. reemplazar emojis por su nombre en texto (para no perder la senal
       de sentimiento que aportan, en vez de solo eliminarlos)
    5. quitar acentos (unidecode) para normalizar antes de quitar puntuacion
    6. eliminar puntuacion y digitos
    7. colapsar espacios
    8. eliminar stopwords en espanol
    """
    if text is None:
        return ""
    t = str(text).lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)
    t = emoji.demojize(t, language="es") if emoji.emoji_count(t) else t
    t = t.replace("_", " ").replace(":", " ")
    t = unidecode(t)
    t = NON_ALPHA_RE.sub(" ", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    tokens = [w for w in t.split() if w not in STOPWORDS_ES and len(w) > 1]
    return " ".join(tokens)


def parse_count_text(value) -> float:
    """Convierte conteos tipo '2,390 vistas' o '1.2K' o '' a numero.

    Reglas documentadas (ver seccion 2.4):
    - valores vacios/NaN -> NaN (no se asume 0 para no perder la distincion
      entre "0 visible" y "dato no observado")
    - se eliminan separadores de miles (comas y puntos usados como tal) y
      texto acompanante (' vistas', ' views')
    - abreviaturas K/M/mil se expanden multiplicando por 1,000 / 1,000,000
    """
    if value is None:
        return float("nan")
    s = str(value).strip().lower()
    if s in ("", "nan", "none"):
        return float("nan")
    s = re.sub(r"(vistas|views|visualizaciones)", "", s).strip()
    mult = 1
    if s.endswith("k"):
        mult = 1_000
        s = s[:-1]
    elif s.endswith("m"):
        mult = 1_000_000
        s = s[:-1]
    elif s.endswith("mil"):
        mult = 1_000
        s = s[:-3]
    s = s.replace(",", "").replace(" ", "")
    if s == "":
        return float("nan")
    try:
        return float(s) * mult
    except ValueError:
        return float("nan")
