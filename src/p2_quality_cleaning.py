"""
Ejercicio 2 - Calidad, limpieza y preprocesamiento.

Genera:
- outputs/tables/02_quality_*.csv           -> diagnostico de calidad (2.1)
- outputs/tables/02_problematic_variables.csv -> variables problematicas (2.2)
- outputs/tables/02_cleaning_effect.csv     -> efecto cuantificado de la limpieza (2.7)
- data/processed/videos_clean.csv           -> videos con IDs normalizados y conteos numericos
- data/processed/comments_clean.csv         -> comentarios con texto_original/texto_limpio,
                                                hashtags, menciones, emojis y conteos numericos
"""
import pandas as pd
import numpy as np

from config import VIDEOS_RAW_CSV, COMMENTS_RAW_CSV, VIDEOS_CLEAN_CSV, COMMENTS_CLEAN_CSV, TABLES_DIR
from text_utils import clean_text, parse_count_text, extract_hashtags, extract_mentions, extract_emojis, parse_list_field


# ---------------------------------------------------------------------------
# 2.1 Diagnostico inicial de calidad
# ---------------------------------------------------------------------------

def quality_overview(df: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "dataset": name,
                "columna": col,
                "dtype_pandas": str(s.dtype),
                "n_faltantes": int(s.isna().sum()),
                "pct_faltantes": round(100 * s.isna().mean(), 2),
                "n_unicos": int(s.nunique(dropna=True)),
                "es_constante": bool(s.nunique(dropna=True) <= 1),
            }
        )
    return pd.DataFrame(rows)


def outlier_report(df: pd.DataFrame, cols: list, name: str) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < low) | (s > high)).sum())
        rows.append(
            {
                "dataset": name,
                "columna": col,
                "min": s.min(),
                "p25": q1,
                "mediana": s.median(),
                "p75": q3,
                "max": s.max(),
                "limite_iqr_inferior": low,
                "limite_iqr_superior": high,
                "n_atipicos_iqr": n_out,
                "pct_atipicos_iqr": round(100 * n_out / len(s), 2),
            }
        )
    return pd.DataFrame(rows)


def consistency_report(videos: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "chequeo": "channel_id -> channel_name unico (videos)",
            "ok": bool((videos.groupby("channel_id")["channel_name"].nunique() <= 1).all()),
        },
        {
            "chequeo": "channel_id -> channel_handle unico (videos)",
            "ok": bool((videos.groupby("channel_id")["channel_handle"].nunique() <= 1).all()),
        },
        {
            "chequeo": "author_channel_id -> author_name unico (comments)",
            "ok": bool((comments.groupby("author_channel_id")["author_name"].nunique() <= 1).all()),
        },
        {
            "chequeo": "channel_id -> channel_name unico (comments)",
            "ok": bool((comments.groupby("channel_id")["channel_name"].nunique() <= 1).all()),
        },
        {
            "chequeo": "upload_date == publish_date (videos)",
            "ok": bool((videos["upload_date"] == videos["publish_date"]).all()),
        },
        {
            "chequeo": "owner_handle == channel_handle (videos)",
            "ok": bool((videos["owner_handle"] == videos["channel_handle"]).all()),
        },
        {
            "chequeo": "todo video_id de comments existe en videos",
            "ok": bool(comments["video_id"].isin(videos["video_id"]).all()),
        },
        {
            "chequeo": "video_id unico en videos (llave primaria)",
            "ok": bool(videos["video_id"].is_unique),
        },
        {
            "chequeo": "comment_id unico en comments (llave primaria)",
            "ok": bool(comments["comment_id"].is_unique),
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2.3 / 2.4 Normalizacion de IDs y conversion de conteos
# ---------------------------------------------------------------------------

ID_COLS_VIDEOS = ["video_id", "channel_id"]
ID_COLS_COMMENTS = ["video_id", "comment_id", "channel_id", "author_channel_id"]


def normalize_ids(df: pd.DataFrame, id_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in id_cols:
        df[col] = df[col].astype(str).str.strip()
    return df


def normalize_names(df: pd.DataFrame, name_cols: list) -> pd.DataFrame:
    """Recorta espacios en nombres/handles visibles. NO reemplaza IDs por nombres."""
    df = df.copy()
    for col in name_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def clean_videos(videos: pd.DataFrame) -> pd.DataFrame:
    v = normalize_ids(videos, ID_COLS_VIDEOS)
    v = normalize_names(v, ["channel_name", "channel_handle", "owner_handle", "title"])

    v["view_count_from_text"] = v["view_count_text"].apply(parse_count_text)
    # view_count (numerico, ya limpio) es la variable recomendada por el enunciado;
    # view_count_from_text solo se calcula para auditar consistencia con view_count_text.
    v["view_count_mismatch"] = (
        v["view_count_from_text"].notna()
        & (v["view_count_from_text"] != v["view_count"])
    )

    v["query_hits_list"] = v["query_hits"].apply(parse_list_field)
    v["keywords_list"] = v["keywords"].apply(parse_list_field)
    v["n_query_hits"] = v["query_hits_list"].apply(len)
    v["n_keywords"] = v["keywords_list"].apply(len)

    v["publish_date"] = pd.to_datetime(v["publish_date"], utc=True, errors="coerce")

    # Columnas redundantes documentadas en 2.2: se conservan pero no se usan en analisis
    return v


def clean_comments(comments: pd.DataFrame) -> pd.DataFrame:
    c = normalize_ids(comments, ID_COLS_COMMENTS)
    c = normalize_names(c, ["channel_name", "author_name", "author_handle"])

    # --- 2.4 conteos ---
    like_missing = c["like_count_text"].astype(str).str.strip().eq("") | c["like_count_text"].isna()
    c["like_count_was_blank"] = like_missing
    # Decision documentada: YouTube no muestra "0" junto a un comentario sin likes,
    # el campo simplemente aparece vacio. Se interpreta blanco == 0 likes visibles,
    # dejando like_count_was_blank como bandera de auditoria de esa decision.
    c["like_count"] = c["like_count_text"].apply(parse_count_text).fillna(0).astype(int)

    # --- 2.5 texto original / limpio ---
    c["texto_original"] = c["text"].astype(str)
    c["hashtags"] = c["texto_original"].apply(extract_hashtags)
    c["mentions"] = c["texto_original"].apply(extract_mentions)
    c["emojis"] = c["texto_original"].apply(extract_emojis)
    c["n_hashtags"] = c["hashtags"].apply(len)
    c["n_mentions"] = c["mentions"].apply(len)
    c["n_emojis"] = c["emojis"].apply(len)
    c["texto_limpio"] = c["texto_original"].apply(clean_text)
    c["texto_limpio_vacio"] = c["texto_limpio"].str.len() == 0

    # is_pinned es constante (False en el 100% de filas) -> se conserva por trazabilidad
    # pero se documenta como sin variabilidad (ver 02_problematic_variables.csv)
    # viewer_rating esta 100% vacia -> se descarta del analisis
    c = c.drop(columns=["viewer_rating"])

    return c


# ---------------------------------------------------------------------------
# 2.7 Efecto cuantificado de la limpieza
# ---------------------------------------------------------------------------

def cleaning_effect(comments_raw: pd.DataFrame, comments_clean: pd.DataFrame) -> pd.DataFrame:
    orig = comments_raw["text"].astype(str)
    clean = comments_clean["texto_limpio"]

    rows = [
        {"metrica": "comentarios totales", "valor": len(comments_raw)},
        {"metrica": "textos originales vacios", "valor": int((orig.str.strip() == "").sum())},
        {"metrica": "textos limpios vacios (todo el contenido era stopword/URL/puntuacion)", "valor": int((clean.str.len() == 0).sum())},
        {"metrica": "duplicados exactos en texto_original", "valor": int(orig.duplicated().sum())},
        {"metrica": "duplicados exactos en texto_limpio", "valor": int(clean.duplicated().sum())},
        {"metrica": "comentarios con al menos un hashtag", "valor": int((comments_clean["n_hashtags"] > 0).sum())},
        {"metrica": "comentarios con al menos una mencion", "valor": int((comments_clean["n_mentions"] > 0).sum())},
        {"metrica": "comentarios con al menos un emoji", "valor": int((comments_clean["n_emojis"] > 0).sum())},
        {"metrica": "caracteres promedio texto_original", "valor": round(orig.str.len().mean(), 1)},
        {"metrica": "caracteres promedio texto_limpio", "valor": round(clean.str.len().mean(), 1)},
    ]
    return pd.DataFrame(rows)


def problematic_variables() -> pd.DataFrame:
    rows = [
        {
            "variable": "viewer_rating (comments)",
            "problema": "100% valores faltantes (406/406)",
            "tratamiento": "Se descarta del analisis; no aporta informacion.",
        },
        {
            "variable": "is_pinned (comments)",
            "problema": "Constante (False en el 100% de los registros)",
            "tratamiento": "Se conserva en el CSV por trazabilidad pero se excluye de analisis por falta de varianza.",
        },
        {
            "variable": "upload_date (videos)",
            "problema": "Coincide con publish_date en el 100% de los casos (redundante)",
            "tratamiento": "Se usa unicamente publish_date en los analisis.",
        },
        {
            "variable": "owner_handle (videos)",
            "problema": "Coincide con channel_handle en el 100% de los casos (redundante)",
            "tratamiento": "Se usa unicamente channel_handle.",
        },
        {
            "variable": "published_time / published_text",
            "problema": "Fecha relativa (ej. 'hace 2 dias'), depende del momento de recoleccion y no es reconstruible a fecha exacta.",
            "tratamiento": "No se convierte a fecha absoluta; se usa publish_date (ISO 8601) de videos para toda cronologia. En comments no existe equivalente exacto, por lo que el orden temporal de comentarios queda como limitacion.",
        },
        {
            "variable": "like_count_text (comments)",
            "problema": "46% de valores en blanco (no es NaN explicito sino cadena vacia)",
            "tratamiento": "Se interpreta como 0 likes visibles (comportamiento normal de la interfaz de YouTube) y se marca con like_count_was_blank para auditoria.",
        },
        {
            "variable": "view_count_text (videos)",
            "problema": "13 valores faltantes; formato de texto localizado ('2,390 vistas')",
            "tratamiento": "Se usa la columna view_count (ya numerica) como fuente principal; view_count_text solo se parsea para verificar consistencia (view_count_mismatch).",
        },
        {
            "variable": "description_snippet (videos)",
            "problema": "Fragmento truncado de la descripcion, redundante y parcial",
            "tratamiento": "Se prioriza description completa para analisis de texto; description_snippet no se usa.",
        },
        {
            "variable": "reply_count (comments)",
            "problema": "No identifica autores de las respuestas (aclaracion explicita del enunciado)",
            "tratamiento": "Se usa unicamente como medida de intensidad de reaccion al comentario principal; nunca como arista autor-autor.",
        },
        {
            "variable": "description (videos)",
            "problema": "1 registro contiene caracteres de reemplazo Unicode (U+FFFD), indicando perdida irreversible de codificacion en el texto original.",
            "tratamiento": "Se conserva el registro (no se elimina), pero se documenta que ese texto especifico no es confiable para analisis de contenido caracter por caracter.",
        },
        {
            "variable": "query_hits / keywords / dataset_sources",
            "problema": "Texto con estructura de lista almacenado como string",
            "tratamiento": "Se parsean con ast.literal_eval a listas reales (query_hits_list, keywords_list) antes de analizarlas.",
        },
    ]
    return pd.DataFrame(rows)


def main():
    videos = pd.read_csv(VIDEOS_RAW_CSV)
    comments = pd.read_csv(COMMENTS_RAW_CSV)

    # 2.1
    q_videos = quality_overview(videos, "youtube_videos")
    q_comments = quality_overview(comments, "youtube_comments")
    pd.concat([q_videos, q_comments]).to_csv(TABLES_DIR / "02_quality_overview.csv", index=False)

    dup_report = pd.DataFrame(
        [
            {"dataset": "youtube_videos", "duplicados_video_id": videos["video_id"].duplicated().sum(), "duplicados_fila_completa": videos.duplicated().sum()},
            {"dataset": "youtube_comments", "duplicados_comment_id": comments["comment_id"].duplicated().sum(), "duplicados_fila_completa": comments.duplicated().sum()},
        ]
    )
    dup_report.to_csv(TABLES_DIR / "02_duplicates.csv", index=False)

    outliers = pd.concat(
        [
            outlier_report(videos, ["view_count"], "youtube_videos"),
            outlier_report(comments, ["reply_count"], "youtube_comments"),
        ]
    )
    outliers.to_csv(TABLES_DIR / "02_outliers_iqr.csv", index=False)

    consistency = consistency_report(videos, comments)
    consistency.to_csv(TABLES_DIR / "02_id_name_consistency.csv", index=False)

    problematic_variables().to_csv(TABLES_DIR / "02_problematic_variables.csv", index=False)

    # 2.3 / 2.4 / 2.5 / 2.6
    videos_clean = clean_videos(videos)
    comments_clean = clean_comments(comments)

    # 2.7
    cleaning_effect(comments, comments_clean).to_csv(TABLES_DIR / "02_cleaning_effect.csv", index=False)

    videos_clean.to_csv(VIDEOS_CLEAN_CSV, index=False)
    comments_clean.to_csv(COMMENTS_CLEAN_CSV, index=False)

    print("=== Diagnostico de calidad (resumen) ===")
    print(f"videos: {videos.shape}, comments: {comments.shape}")
    print(f"duplicados video_id: {videos['video_id'].duplicated().sum()}, duplicados comment_id: {comments['comment_id'].duplicated().sum()}")
    print(f"like_count_text en blanco: {int((comments['like_count_text'].astype(str).str.strip()=='').sum())} / {len(comments)}")
    print(f"consistencia IDs/nombres OK en todos los chequeos: {consistency['ok'].all()}")
    print("\nArchivos generados en outputs/tables/02_*.csv y data/processed/*_clean.csv")


if __name__ == "__main__":
    main()
