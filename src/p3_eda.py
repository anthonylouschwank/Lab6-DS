"""
Ejercicio 3 - Analisis exploratorio.

Genera tablas (outputs/tables/03_*.csv) y figuras (outputs/figures/03_*.png)
para describir videos, canales, comentarios y autores; concentracion de la
participacion; relacion popularidad-participacion; y frecuencias de
palabras/bigramas/hashtags. Los valores numericos clave se imprimen en
stdout para poder citarlos con precision en el informe.
"""
import ast
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud

from config import VIDEOS_CLEAN_CSV, COMMENTS_CLEAN_CSV, TABLES_DIR, FIGURES_DIR

plt.rcParams["figure.dpi"] = 130


def savefig(fig, name):
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"figura guardada: {path.name}")


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    cum = np.cumsum(x)
    return (n + 1 - 2 * (cum.sum() / cum[-1])) / n


def basic_counts(videos, comments) -> pd.DataFrame:
    rows = [
        {"metrica": "videos totales", "valor": len(videos)},
        {"metrica": "canales unicos (videos)", "valor": videos["channel_id"].nunique()},
        {"metrica": "comentarios totales", "valor": len(comments)},
        {"metrica": "autores unicos (comentarios)", "valor": comments["author_channel_id"].nunique()},
        {"metrica": "videos con al menos un comentario", "valor": comments["video_id"].nunique()},
        {"metrica": "canales con al menos un video comentado", "valor": comments["channel_id"].nunique()},
        {"metrica": "categorias distintas de video", "valor": videos["category"].nunique()},
        {"metrica": "consultas de busqueda distintas (source_query, videos)", "valor": videos["source_query"].nunique()},
    ]
    return pd.DataFrame(rows)


def per_video_per_channel(videos, comments) -> tuple[pd.DataFrame, pd.DataFrame]:
    videos_per_channel = videos.groupby(["channel_id", "channel_name"]).size().reset_index(name="n_videos")
    videos_per_channel = videos_per_channel.sort_values("n_videos", ascending=False)

    per_video = comments.groupby(["video_id", "video_title"]).agg(
        n_comentarios=("comment_id", "count"),
        n_autores_unicos=("author_channel_id", "nunique"),
        likes_totales=("like_count", "sum"),
        respuestas_totales=("reply_count", "sum"),
    ).reset_index()
    per_video = per_video.merge(videos[["video_id", "channel_name", "view_count", "category"]], on="video_id", how="left")
    per_video = per_video.sort_values("n_comentarios", ascending=False)
    return videos_per_channel, per_video


def numeric_summaries(videos, comments) -> pd.DataFrame:
    parts = []
    for col, df, label in [
        ("view_count", videos, "videos.view_count"),
        ("reply_count", comments, "comments.reply_count"),
        ("like_count", comments, "comments.like_count"),
    ]:
        s = df[col]
        parts.append(
            {
                "variable": label,
                "n": s.count(),
                "media": round(s.mean(), 2),
                "mediana": s.median(),
                "std": round(s.std(), 2),
                "min": s.min(),
                "max": s.max(),
            }
        )
    return pd.DataFrame(parts)


def category_and_query_tables(videos, comments):
    cat_videos = videos["category"].value_counts().reset_index()
    cat_videos.columns = ["category", "n_videos"]

    cat_comments = comments.merge(videos[["video_id", "category"]], on="video_id", how="left")
    cat_comments = cat_comments["category"].value_counts().reset_index()
    cat_comments.columns = ["category", "n_comentarios"]

    query_videos = videos["source_query"].value_counts().reset_index()
    query_videos.columns = ["source_query", "n_videos"]

    return cat_videos, cat_comments, query_videos


def word_bigram_frequencies(comments, top_n=20):
    tokens_per_comment = comments["texto_limpio"].fillna("").str.split()
    all_tokens = [t for toks in tokens_per_comment for t in toks]
    word_freq = Counter(all_tokens).most_common(top_n)

    bigrams = []
    for toks in tokens_per_comment:
        bigrams.extend(zip(toks, toks[1:]))
    bigram_freq = Counter([" ".join(b) for b in bigrams]).most_common(top_n)

    return (
        pd.DataFrame(word_freq, columns=["palabra", "frecuencia"]),
        pd.DataFrame(bigram_freq, columns=["bigrama", "frecuencia"]),
        all_tokens,
    )


def hashtag_keyword_frequencies(videos, comments, top_n=20):
    hashtags = Counter()
    for lst in comments["hashtags"].apply(ast.literal_eval):
        hashtags.update(lst)

    keywords = Counter()
    for lst in videos["keywords_list"].apply(ast.literal_eval):
        keywords.update([k.lower() for k in lst])

    return (
        pd.DataFrame(hashtags.most_common(top_n), columns=["hashtag", "frecuencia"]),
        pd.DataFrame(keywords.most_common(top_n), columns=["keyword", "frecuencia"]),
    )


def concentration_analysis(per_video: pd.DataFrame, comments: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    total = per_video["n_comentarios"].sum()
    top1 = per_video["n_comentarios"].iloc[0]
    top3 = per_video["n_comentarios"].iloc[:3].sum()
    top5 = per_video["n_comentarios"].iloc[:5].sum()

    by_channel = comments.groupby("channel_name")["comment_id"].count().sort_values(ascending=False)
    top1_ch = by_channel.iloc[0]
    top3_ch = by_channel.iloc[:3].sum()

    rows = [
        {"metrica": "video mas comentado (%)", "valor": round(100 * top1 / total, 1)},
        {"metrica": "top 3 videos (%)", "valor": round(100 * top3 / total, 1)},
        {"metrica": "top 5 videos (%)", "valor": round(100 * top5 / total, 1)},
        {"metrica": "canal con mas comentarios (%)", "valor": round(100 * top1_ch / total, 1)},
        {"metrica": "top 3 canales (%)", "valor": round(100 * top3_ch / total, 1)},
        {"metrica": "gini comentarios por video (de los 19 con comentarios)", "valor": round(gini(per_video["n_comentarios"].values), 3)},
        {"metrica": "gini vistas por video (293 videos)", "valor": round(gini(videos["view_count"].values), 3)},
    ]
    return pd.DataFrame(rows)


def popularity_vs_participation(per_video: pd.DataFrame):
    """Correlacion de Spearman (rangos) sin depender de scipy."""
    x = pd.Series(per_video["view_count"].values).rank()
    y = pd.Series(per_video["n_comentarios"].values).rank()
    rho = x.corr(y, method="pearson")
    return rho, None


def main():
    videos = pd.read_csv(VIDEOS_CLEAN_CSV)
    comments = pd.read_csv(COMMENTS_CLEAN_CSV)

    counts = basic_counts(videos, comments)
    counts.to_csv(TABLES_DIR / "03_basic_counts.csv", index=False)
    print(counts.to_string(index=False))

    videos_per_channel, per_video = per_video_per_channel(videos, comments)
    videos_per_channel.to_csv(TABLES_DIR / "03_videos_per_channel.csv", index=False)
    per_video.to_csv(TABLES_DIR / "03_comments_authors_per_video.csv", index=False)

    numsum = numeric_summaries(videos, comments)
    numsum.to_csv(TABLES_DIR / "03_numeric_summaries.csv", index=False)
    print("\n" + numsum.to_string(index=False))

    cat_videos, cat_comments, query_videos = category_and_query_tables(videos, comments)
    cat_videos.to_csv(TABLES_DIR / "03_category_videos.csv", index=False)
    cat_comments.to_csv(TABLES_DIR / "03_category_comments.csv", index=False)
    query_videos.to_csv(TABLES_DIR / "03_source_query_videos.csv", index=False)

    word_freq, bigram_freq, all_tokens = word_bigram_frequencies(comments)
    word_freq.to_csv(TABLES_DIR / "03_word_frequency.csv", index=False)
    bigram_freq.to_csv(TABLES_DIR / "03_bigram_frequency.csv", index=False)

    hashtag_freq, keyword_freq = hashtag_keyword_frequencies(videos, comments)
    hashtag_freq.to_csv(TABLES_DIR / "03_hashtag_frequency.csv", index=False)
    keyword_freq.to_csv(TABLES_DIR / "03_keyword_frequency.csv", index=False)

    concentration = concentration_analysis(per_video, comments, videos)
    concentration.to_csv(TABLES_DIR / "03_concentration.csv", index=False)
    print("\n" + concentration.to_string(index=False))

    rho, p = popularity_vs_participation(per_video)
    print(f"\nCorrelacion Spearman vistas vs n_comentarios (19 videos con comentarios): rho={rho:.3f}")
    pd.DataFrame([{"spearman_rho": rho, "p_value": p, "n": len(per_video)}]).to_csv(
        TABLES_DIR / "03_popularity_vs_participation.csv", index=False
    )

    # --- Figuras ---
    fig, ax = plt.subplots(figsize=(8, 5))
    per_video.set_index("video_title")["n_comentarios"].sort_values().plot.barh(ax=ax)
    ax.set_xlabel("Numero de comentarios")
    ax.set_ylabel("")
    ax.set_title("Comentarios por video (19 videos con comentarios)")
    savefig(fig, "03_comentarios_por_video.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    comments.groupby("channel_name")["comment_id"].count().sort_values().plot.barh(ax=ax)
    ax.set_xlabel("Numero de comentarios")
    ax.set_title("Comentarios por canal")
    savefig(fig, "03_comentarios_por_canal.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    cat_videos.set_index("category")["n_videos"].sort_values().plot.barh(ax=ax)
    ax.set_xlabel("Numero de videos")
    ax.set_title("Videos por categoria (293 videos)")
    savefig(fig, "03_videos_por_categoria.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(np.log10(videos["view_count"].clip(lower=1)), bins=30)
    ax.set_xlabel("log10(view_count)")
    ax.set_ylabel("Numero de videos")
    ax.set_title("Distribucion de visualizaciones (escala log)")
    savefig(fig, "03_histograma_views.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(per_video["view_count"], per_video["n_comentarios"])
    for _, r in per_video.iterrows():
        ax.annotate(r["video_id"], (r["view_count"], r["n_comentarios"]), fontsize=6, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Visualizaciones (log)")
    ax.set_ylabel("Numero de comentarios en la muestra")
    ax.set_title("Popularidad vs. participacion (19 videos con comentarios)")
    savefig(fig, "03_popularidad_vs_participacion.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    word_freq.set_index("palabra")["frecuencia"].sort_values().plot.barh(ax=ax)
    ax.set_xlabel("Frecuencia")
    ax.set_title("Top 20 palabras mas frecuentes (texto_limpio)")
    savefig(fig, "03_top_palabras.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    bigram_freq.set_index("bigrama")["frecuencia"].sort_values().plot.barh(ax=ax)
    ax.set_xlabel("Frecuencia")
    ax.set_title("Top 20 bigramas mas frecuentes (texto_limpio)")
    savefig(fig, "03_top_bigramas.png")

    wc = WordCloud(width=1000, height=600, background_color="white", colormap="viridis")
    wc.generate(" ".join(all_tokens))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Nube de palabras - comentarios (texto_limpio)")
    savefig(fig, "03_wordcloud.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    keyword_freq.set_index("keyword")["frecuencia"].sort_values().plot.barh(ax=ax)
    ax.set_xlabel("Frecuencia (en videos)")
    ax.set_title("Top 20 keywords de video mas frecuentes")
    savefig(fig, "03_top_keywords_videos.png")

    print("\nTablas y figuras del EDA generadas en outputs/tables y outputs/figures (prefijo 03_).")


if __name__ == "__main__":
    main()
