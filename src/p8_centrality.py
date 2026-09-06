"""Ejercicio 8: centralidad, participantes recurrentes y nodos puente.

Se calculan cuatro medidas complementarias sobre la bipartita observada
(autor-video) y, para videos, tambien sobre la proyeccion video-video:

- grado (degree_centrality): alcance directo, normalizado por N-1.
- fuerza (grado ponderado por 'weight'): intensidad de la participacion
  (comentarios totales en autores; comentarios/autores compartidos en
  videos), no solo su alcance.
- intermediacion (betweenness_centrality, sin pesos): que tan seguido un
  nodo se ubica en el camino mas corto entre otros dos nodos. Es la medida
  mas directa de "puente": un autor con intermediacion > 0 conecta al
  menos dos videos que de otro modo quedarian en componentes separadas.
- PageRank (ponderado por 'weight'): importancia recursiva -- un nodo es
  importante si esta conectado a otros nodos importantes, ponderado por
  la intensidad de comentarios. Complementa al grado, que solo ve vecinos
  directos.

No se usa cercania (closeness) como medida principal porque la red tiene
10 componentes conexas (ver Ejercicio 6): la cercania solo es comparable
dentro de una misma componente y su promedio global se distorsiona por el
tamano dispar de las 19 "estrellas" autor-video.

Los puntos de articulacion (nx.articulation_points) dan la respuesta exacta
a "si elimino este nodo, la red se segmenta": son nodos cuya remocion
aumenta el numero de componentes conexas de su componente original. En una
bipartita compuesta por estrellas (un video con sus autores) casi todo
punto de articulacion coincide con un nodo de intermediacion > 0, salvo
casos de estrellas con un unico autor (donde el autor tambien es un punto
de articulacion trivial, pero no es un "puente" real entre contenidos).

Fuentes:
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.articulation_points.html
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from config import NETWORK_DIR, FIGURES_DIR, TABLES_DIR
from p5_network_projections import load_bipartite


def compute_centralities(graph, weight='weight'):
    degree = nx.degree_centrality(graph)
    strength = dict(graph.degree(weight=weight))
    betweenness = nx.betweenness_centrality(graph, weight=None, normalized=True)
    pagerank = nx.pagerank(graph, weight=weight)
    assert set(degree) == set(strength) == set(betweenness) == set(pagerank) == set(graph)
    articulation = set(nx.articulation_points(graph))
    rows = []
    for n, d in graph.nodes(data=True):
        row = {'node_id': n, **d, 'degree_centrality': degree[n], 'strength': strength[n],
               'betweenness_centrality': betweenness[n], 'pagerank': pagerank[n],
               'is_articulation_point': n in articulation}
        rows.append(row)
    table = pd.DataFrame(rows)
    assert len(table) == graph.number_of_nodes()
    return table, articulation


def plot_top(table, value_col, label_col, title, filename, n=15):
    top = table.sort_values(value_col, ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top[label_col].str.slice(0, 45)[::-1], top[value_col][::-1])
    ax.set_xlabel(value_col)
    ax.set_title(title)
    fig.savefig(FIGURES_DIR / filename, bbox_inches='tight', dpi=150)
    plt.close(fig)


def main():
    bipartite = load_bipartite()
    table, articulation = compute_centralities(bipartite)

    authors = table[table.bipartite == 'author'].copy()
    videos = table[table.bipartite == 'video'].copy()
    assert len(authors) == 332 and len(videos) == 19

    authors = authors.sort_values(['betweenness_centrality', 'strength'], ascending=False)
    videos = videos.sort_values(['betweenness_centrality', 'pagerank'], ascending=False)

    authors.to_csv(TABLES_DIR / '08_author_centrality.csv', index=False)
    videos.to_csv(TABLES_DIR / '08_video_centrality.csv', index=False)

    # Puntos de articulacion, separados por tipo de nodo.
    articulation_table = table[table.is_articulation_point][
        ['node_id', 'bipartite', 'label', 'betweenness_centrality']
    ].sort_values('betweenness_centrality', ascending=False)
    articulation_table.to_csv(TABLES_DIR / '08_articulation_points.csv', index=False)

    bridge_authors = authors[authors.betweenness_centrality > 0]
    assert (bridge_authors.n_videos_distintos > 1).all()
    articulator_videos = videos[videos.betweenness_centrality > 0]

    # Centralidad tambien sobre la proyeccion video-video: mide directamente
    # "capacidad de conectar audiencias" (una arista ya es audiencia
    # compartida), a diferencia del grado en la bipartita, que solo cuenta
    # autores propios sin decir si esos autores tambien comentan en otros
    # videos.
    video_projection = nx.read_graphml(NETWORK_DIR / 'video_projection.graphml')
    proj_table, proj_articulation = compute_centralities(video_projection)
    proj_table = proj_table.sort_values('betweenness_centrality', ascending=False)
    proj_table.to_csv(TABLES_DIR / '08_video_projection_centrality.csv', index=False)

    plot_top(authors, 'betweenness_centrality', 'label', 'Top autores por intermediacion (puentes)', '08_top_betweenness_authors.png')
    plot_top(videos, 'pagerank', 'label', 'Top videos por PageRank (bipartita)', '08_top_pagerank_videos.png')
    plot_top(proj_table, 'betweenness_centrality', 'label', 'Top videos por intermediacion (proyeccion video-video)', '08_top_betweenness_videos_projection.png')

    print(f'Autores puente (betweenness > 0 en la bipartita): {len(bridge_authors)} de {len(authors)}')
    print(f'Videos articuladores (betweenness > 0 en la bipartita): {len(articulator_videos)} de {len(videos)}')
    print(f'Puntos de articulacion totales en la bipartita: {len(articulation)} '
          f'({sum(table.loc[table.node_id.isin(articulation), "bipartite"] == "author")} autores, '
          f'{sum(table.loc[table.node_id.isin(articulation), "bipartite"] == "video")} videos)')
    print('\nTop 5 autores puente (intermediacion):')
    print(authors[['node_id', 'label', 'n_videos_distintos', 'n_comentarios_totales', 'betweenness_centrality']].head(5).to_string(index=False))
    print('\nRanking de videos (bipartita) por PageRank / intermediacion:')
    print(videos[['node_id', 'label', 'n_autores_unicos', 'n_comentarios', 'pagerank', 'betweenness_centrality']].to_string(index=False))


if __name__ == '__main__':
    main()
