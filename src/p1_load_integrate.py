"""
Ejercicio 1 - Carga, comprension e integracion de los datos.

Carga youtube_videos.csv y youtube_comments.csv, verifica las llaves
primarias/foraneas y construye la tabla integrada comentario+video mediante
video_id. Guarda la tabla integrada en data/processed/ y un resumen de
integracion en outputs/tables/.
"""
import pandas as pd

from config import VIDEOS_RAW_CSV, COMMENTS_RAW_CSV, MERGED_CSV, TABLES_DIR


def load_raw():
    videos = pd.read_csv(VIDEOS_RAW_CSV)
    comments = pd.read_csv(COMMENTS_RAW_CSV)
    return videos, comments


def check_primary_keys(videos: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "tabla": "youtube_videos",
            "llave_primaria": "video_id",
            "n_filas": len(videos),
            "n_valores_unicos": videos["video_id"].nunique(),
            "duplicados": videos["video_id"].duplicated().sum(),
        },
        {
            "tabla": "youtube_comments",
            "llave_primaria": "comment_id",
            "n_filas": len(comments),
            "n_valores_unicos": comments["comment_id"].nunique(),
            "duplicados": comments["comment_id"].duplicated().sum(),
        },
    ]
    return pd.DataFrame(rows)


def integrate(videos: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    """Une comentarios con sus videos mediante video_id (llave foranea).

    Se usa left join desde comments porque el objetivo es analizar
    comentarios en su contexto de video; todo comentario debe tener un
    video asociado (video_id es obligatorio en la recoleccion).
    """
    merged = comments.merge(
        videos,
        on="video_id",
        how="left",
        suffixes=("_comment", "_video"),
        indicator=True,
    )
    return merged


def integration_summary(videos: pd.DataFrame, comments: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    matched = (merged["_merge"] == "both").sum()
    unmatched = (merged["_merge"] == "left_only").sum()
    videos_con_comentarios = comments["video_id"].nunique()
    videos_sin_comentarios = videos["video_id"].nunique() - videos_con_comentarios

    rows = [
        {"metrica": "comentarios totales", "valor": len(comments)},
        {"metrica": "comentarios asociados a un video (match)", "valor": matched},
        {"metrica": "comentarios sin video asociado (huerfanos)", "valor": unmatched},
        {"metrica": "videos totales", "valor": len(videos)},
        {"metrica": "videos con >=1 comentario en la muestra", "valor": videos_con_comentarios},
        {"metrica": "videos sin comentarios en la muestra", "valor": videos_sin_comentarios},
        {
            "metrica": "porcentaje de videos cubiertos por comentarios",
            "valor": round(100 * videos_con_comentarios / len(videos), 2),
        },
    ]
    return pd.DataFrame(rows)


def main():
    videos, comments = load_raw()

    pk_report = check_primary_keys(videos, comments)
    pk_report.to_csv(TABLES_DIR / "01_primary_key_check.csv", index=False)
    print("=== Verificacion de llaves primarias ===")
    print(pk_report.to_string(index=False))

    merged = integrate(videos, comments)
    summary = integration_summary(videos, comments, merged)
    summary.to_csv(TABLES_DIR / "01_integration_summary.csv", index=False)
    print("\n=== Resumen de integracion (video_id) ===")
    print(summary.to_string(index=False))

    merged = merged.drop(columns=["_merge"])
    merged.to_csv(MERGED_CSV, index=False)
    print(f"\nTabla integrada guardada en: {MERGED_CSV}")


if __name__ == "__main__":
    main()
