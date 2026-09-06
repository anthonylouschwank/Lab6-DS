"""Ejercicio 9: analisis de sentimiento de los comentarios.

Herramienta: pysentimiento (analyzer.predict, task='sentiment', lang='es'),
que usa RoBERTuito -- un modelo RoBERTa preentrenado y afinado sobre tuits
en espanol (Perez et al., 2021, https://arxiv.org/abs/2111.09453). Se elige
sobre un lexico simple (tipo VADER/iSOL) porque nuestros comentarios son
cortos, informales, con groserias, emojis convertidos a texto y errores de
tipeo -- exactamente el dominio (redes sociales en espanol) para el que
RoBERTuito fue entrenado, y porque un modelo entrenado capta negacion y
contexto ("no es bueno") mejor que sumar polaridad palabra por palabra.

Se analiza texto_original (no texto_limpio): quitar stopwords, acentos y
emojis antes de clasificar borraria justamente las senales que un modelo de
lenguaje usa (orden de palabras, tildes, el emoji original). texto_limpio
solo se usa para las palabras/bigramas frecuentes (Ejercicios 3 y 7).

Salida de cada comentario: etiqueta discreta (NEG/NEU/POS) y probabilidad
de cada clase; se reporta la etiqueta y tambien el promedio de probabilidad
POS-NEG como score continuo para comparar grupos.
"""
import pandas as pd
from pysentimiento import create_analyzer

from config import TABLES_DIR, FIGURES_DIR, COMMENTS_CLEAN_CSV, NETWORK_DIR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MIN_GROUP_SIZE = 5  # tamano minimo de muestra para reportar una comparacion de sentimiento


def analyze_sentiment(texts: list[str]) -> pd.DataFrame:
    analyzer = create_analyzer(task='sentiment', lang='es')
    outputs = analyzer.predict(texts)
    rows = [
        {
            'sentiment': o.output,
            'proba_neg': o.probas['NEG'],
            'proba_neu': o.probas['NEU'],
            'proba_pos': o.probas['POS'],
            'sentiment_score': o.probas['POS'] - o.probas['NEG'],
        }
        for o in outputs
    ]
    table = pd.DataFrame(rows)
    assert len(table) == len(texts)
    assert table['sentiment'].isin(['NEG', 'NEU', 'POS']).all()
    return table


def group_summary(df: pd.DataFrame, group_col: str, label_col: str | None = None) -> pd.DataFrame:
    counts = df.groupby(group_col).size().rename('n_comentarios')
    rows = df.groupby(group_col).agg(
        pct_negativo=('sentiment', lambda s: 100 * (s == 'NEG').mean()),
        pct_neutro=('sentiment', lambda s: 100 * (s == 'NEU').mean()),
        pct_positivo=('sentiment', lambda s: 100 * (s == 'POS').mean()),
        score_promedio=('sentiment_score', 'mean'),
    )
    out = rows.join(counts).reset_index()
    if label_col and label_col in df.columns:
        labels = df.drop_duplicates(group_col).set_index(group_col)[label_col]
        out[label_col] = out[group_col].map(labels)
    out['muestra_suficiente'] = out['n_comentarios'] >= MIN_GROUP_SIZE
    return out.sort_values('n_comentarios', ascending=False)


def main():
    comments = pd.read_csv(COMMENTS_CLEAN_CSV, keep_default_na=False)

    sentiment = analyze_sentiment(comments['texto_original'].tolist())
    comments = pd.concat([comments.reset_index(drop=True), sentiment], axis=1)

    overall = comments['sentiment'].value_counts(normalize=True).mul(100).round(1)
    print('Distribucion global de sentimiento (%):')
    print(overall.to_string())

    detail_cols = [
        'comment_id', 'video_id', 'video_title', 'channel_name', 'author_channel_id',
        'texto_original', 'sentiment', 'proba_neg', 'proba_neu', 'proba_pos', 'sentiment_score',
    ]
    comments[detail_cols].to_csv(TABLES_DIR / '09_comment_sentiment.csv', index=False)

    # --- 9.2 comparaciones ---
    by_video = group_summary(comments, 'video_id', 'video_title')
    by_video.to_csv(TABLES_DIR / '09_sentiment_by_video.csv', index=False)

    by_channel = group_summary(comments, 'channel_name')
    by_channel.to_csv(TABLES_DIR / '09_sentiment_by_channel.csv', index=False)

    by_category = comments.merge(
        pd.read_csv(NETWORK_DIR / 'bipartite_nodes.csv')[['node_id', 'category']].rename(columns={'node_id': 'video_id'}),
        on='video_id', how='left',
    )
    by_category = group_summary(by_category, 'category')
    by_category.to_csv(TABLES_DIR / '09_sentiment_by_category.csv', index=False)

    community_map = pd.read_csv(NETWORK_DIR / '07_video_communities.csv')[['video_id', 'community_id']]
    with_community = comments.merge(community_map, on='video_id', how='left')
    assert with_community['community_id'].notna().all()
    by_community = group_summary(with_community, 'community_id')
    by_community.to_csv(TABLES_DIR / '09_sentiment_by_community.csv', index=False)

    # Completa el archivo que dejo pendiente el Ejercicio 7 (mismo esquema:
    # comment_id, video_id, author_channel_id, community_id, sentiment).
    placeholder = pd.read_csv(NETWORK_DIR / '07_comment_community_sentiment.csv')
    updated = placeholder.drop(columns=['sentiment', 'sentiment_status']).merge(
        comments[['comment_id', 'sentiment']], on='comment_id', how='left'
    )
    updated['sentiment_status'] = 'calculado_ejercicio_9_pysentimiento_robertuito'
    assert len(updated) == len(placeholder) and updated['sentiment'].notna().all()
    updated.to_csv(NETWORK_DIR / '07_comment_community_sentiment.csv', index=False)

    # --- Figuras ---
    fig, ax = plt.subplots(figsize=(5, 4))
    order = ['NEG', 'NEU', 'POS']
    counts = comments['sentiment'].value_counts().reindex(order)
    ax.bar(order, counts.values, color=['#c0392b', '#95a5a6', '#27ae60'])
    ax.set_ylabel('Numero de comentarios')
    ax.set_title(f'Sentimiento global (n={len(comments)})')
    fig.savefig(FIGURES_DIR / '09_sentiment_global.png', bbox_inches='tight', dpi=150)
    plt.close(fig)

    top_videos = by_video[by_video['muestra_suficiente']].sort_values('n_comentarios', ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = top_videos.set_index('video_title')[['pct_negativo', 'pct_neutro', 'pct_positivo']]
    bars.iloc[::-1].plot.barh(stacked=True, ax=ax, color=['#c0392b', '#95a5a6', '#27ae60'])
    ax.set_xlabel('% de comentarios')
    ax.set_title(f'Sentimiento por video (videos con >= {MIN_GROUP_SIZE} comentarios)')
    ax.legend(['Negativo', 'Neutro', 'Positivo'], loc='lower right', fontsize=7)
    fig.savefig(FIGURES_DIR / '09_sentiment_by_video.png', bbox_inches='tight', dpi=150)
    plt.close(fig)

    top_communities = by_community[by_community['muestra_suficiente']].sort_values('community_id')
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = top_communities.set_index('community_id')[['pct_negativo', 'pct_neutro', 'pct_positivo']]
    bars.plot.bar(stacked=True, ax=ax, color=['#c0392b', '#95a5a6', '#27ae60'])
    ax.set_ylabel('% de comentarios')
    ax.set_xlabel('Comunidad (Ejercicio 7)')
    ax.set_title(f'Sentimiento por comunidad (comunidades con >= {MIN_GROUP_SIZE} comentarios)')
    ax.legend(['Negativo', 'Neutro', 'Positivo'], fontsize=7)
    fig.savefig(FIGURES_DIR / '09_sentiment_by_community.png', bbox_inches='tight', dpi=150)
    plt.close(fig)

    print('\nSentimiento por video (muestra suficiente, >= %d comentarios):' % MIN_GROUP_SIZE)
    print(top_videos[['video_title', 'n_comentarios', 'pct_negativo', 'pct_neutro', 'pct_positivo', 'score_promedio']].to_string(index=False))
    print('\nSentimiento por comunidad (muestra suficiente):')
    print(top_communities[['community_id', 'n_comentarios', 'pct_negativo', 'pct_neutro', 'pct_positivo', 'score_promedio']].to_string(index=False))
    print('\nSentimiento por canal:')
    print(by_channel[['channel_name', 'n_comentarios', 'pct_negativo', 'pct_neutro', 'pct_positivo', 'score_promedio']].to_string(index=False))


if __name__ == '__main__':
    main()
