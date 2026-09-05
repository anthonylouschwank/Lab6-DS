"""Ejercicio 7: comunidades de videos con audiencia observada compartida.

Se elige la proyeccion video–video ponderada porque agrupa contenidos por
participantes comunes. Louvain optimiza modularidad de forma heuristica:
compara peso interno observado con el esperado bajo un modelo nulo que
preserva fuerzas (grados ponderados). No garantiza un optimo global.
weight cuenta autores distintos compartidos; resolucion=1 y semilla fija.
Las comunidades son estructurales en esta muestra, no grupos sociales
confirmados. Aislados permanecen como comunidades unitarias; sin aristas
la modularidad queda indefinida. Sentimiento pendiente del ejercicio 9.
Fuente: https://networkx.org/documentation/stable/reference/algorithms/generated/
networkx.algorithms.community.louvain.louvain_communities.html
"""
import json

import networkx as nx
import pandas as pd

from config import NETWORK_DIR, VIDEOS_CLEAN_CSV, COMMENTS_CLEAN_CSV, RANDOM_SEED
from p3_eda import word_bigram_frequencies, hashtag_keyword_frequencies
from p5_network_projections import plot_projection


def detect_communities(graph):
    if graph.number_of_edges():
        partition = nx.community.louvain_communities(graph, weight='weight', resolution=1, seed=RANDOM_SEED)
        modularity = nx.community.modularity(graph, partition, weight='weight', resolution=1)
    else:
        partition, modularity = [{n} for n in graph], float('nan')
    partition = sorted(partition, key=lambda c: (-len(c), min(c)))
    assert sum(map(len, partition)) == len(graph)
    assert set().union(*partition) == set(graph) if partition else len(graph) == 0
    mapping = {n: cid for cid, nodes in enumerate(partition, 1) for n in nodes}
    assert len(mapping) == len(graph)
    return mapping, modularity


def main():
    graph = nx.read_graphml(NETWORK_DIR / 'video_projection.graphml')
    videos = pd.read_csv(VIDEOS_CLEAN_CSV, keep_default_na=False)
    comments = pd.read_csv(COMMENTS_CLEAN_CSV, keep_default_na=False)
    assert set(graph) == set(comments.video_id)
    mapping, modularity = detect_communities(graph)
    members = pd.DataFrame([dict(video_id=n, community_id=mapping[n], **d) for n, d in graph.nodes(data=True)])
    members.to_csv(NETWORK_DIR / '07_video_communities.csv', index=False)
    joined = comments.assign(community_id=comments.video_id.map(mapping))
    assert joined.community_id.notna().all() and joined.comment_id.is_unique
    joined[['comment_id', 'video_id', 'author_channel_id', 'community_id']].assign(
        sentiment=pd.NA, sentiment_status='pendiente_ejercicio_9').to_csv(
        NETWORK_DIR / '07_comment_community_sentiment.csv', index=False)
    summary, authors = [], []
    for cid, subset in members.groupby('community_id'):
        observed = joined[joined.community_id == cid]
        subgraph = graph.subgraph(subset.video_id)
        channels = subset[['channel_id', 'channel_name']].drop_duplicates().sort_values('channel_id')
        summary.append(dict(community_id=cid, n_videos=len(subset), n_channels=channels.channel_id.nunique(),
                            n_authors=observed.author_channel_id.nunique(), n_comments=len(observed),
                            share_comments=len(observed) / len(comments),
                            comments_per_author=len(observed) / observed.author_channel_id.nunique(),
                            comments_per_video=len(observed) / len(subset),
                            internal_shared_author_weight=subgraph.size(weight='weight'),
                            channels=json.dumps(channels.to_dict('records'), ensure_ascii=False),
                            sentiment=pd.NA, sentiment_status='pendiente_ejercicio_9'))
        grouped = observed.groupby('author_channel_id').agg(author_name=('author_name', 'first'),
                    n_comments=('comment_id', 'size'), n_videos=('video_id', 'nunique')).reset_index()
        grouped.insert(0, 'community_id', cid)
        authors.append(grouped)
    summary = pd.DataFrame(summary).sort_values(['n_comments', 'n_videos', 'community_id'], ascending=[False, False, True])
    selected = summary.head(3).community_id.tolist()
    summary['selected_for_characterization'] = summary.community_id.isin(selected)
    assert summary.n_comments.sum() == len(comments) and abs(summary.share_comments.sum() - 1) < 1e-10
    # Autores pueden aparecer en varias comunidades: no sumar n_authors como unicos globales.
    summary.to_csv(NETWORK_DIR / '07_community_summary.csv', index=False)
    pd.concat(authors).sort_values(['community_id', 'n_comments', 'author_channel_id'], ascending=[True, False, True]).to_csv(
        NETWORK_DIR / '07_community_authors.csv', index=False)
    members.sort_values(['community_id', 'n_comentarios', 'video_id'], ascending=[True, False, True]).to_csv(
        NETWORK_DIR / '07_community_members.csv', index=False)
    terms = []
    for cid in selected:
        observed = joined[joined.community_id == cid]
        selected_videos = videos[videos.video_id.isin(observed.video_id)]
        words, bigrams, _ = word_bigram_frequencies(observed)
        hashtags, keywords = hashtag_keyword_frequencies(selected_videos, observed)
        for kind, table in [('word', words), ('bigram', bigrams), ('hashtag', hashtags), ('keyword', keywords)]:
            for term, frequency in table.itertuples(index=False, name=None):
                terms.append(dict(community_id=cid, term_type=kind, term=term, frequency=frequency))
    pd.DataFrame(terms, columns=['community_id', 'term_type', 'term', 'frequency']).to_csv(
        NETWORK_DIR / '07_community_terms.csv', index=False)
    pd.DataFrame([dict(network='video_projection', algorithm='networkx_louvain', networkx_version=nx.__version__,
                       seed=RANDOM_SEED, resolution=1, weight='weight', modularity=modularity,
                       n_communities=len(summary), n_characterized=len(selected),
                       fewer_than_three=len(summary) < 3, selection='n_comments_desc_n_videos_desc_id_asc')]).to_csv(
        NETWORK_DIR / '07_community_metrics.csv', index=False)
    nx.set_node_attributes(graph, mapping, 'community_id')
    nx.write_graphml(graph, NETWORK_DIR / '07_video_communities.graphml')
    plot_projection(graph, '07_video_communities.png', 'Comunidades de videos: audiencia compartida', mapping)
    print(summary[['community_id', 'n_videos', 'n_channels', 'n_authors', 'n_comments', 'share_comments']].to_string(index=False))
    print(f'Louvain ponderado: {len(summary)} comunidades; modularidad={modularity:.4f}.')
    print(f'Caracterizacion por participacion: {selected}. Sentimiento pendiente del ejercicio 9.')


if __name__ == '__main__':
    main()
