"""Ejercicio 5: proyecciones ponderadas de la bipartita observada.

Autor–autor significa co-participacion, no amistad, acuerdo o conversacion.
Video–video significa audiencia compartida, no similitud semantica.
Los pesos de comentarios de la bipartita NO se trasladan a las proyecciones.
"""
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from config import NETWORK_DIR, FIGURES_DIR, COMMENTS_CLEAN_CSV, RANDOM_SEED


def load_bipartite():
    graph = nx.read_graphml(NETWORK_DIR / 'bipartite_author_video.graphml')
    assert not graph.is_directed() and not graph.is_multigraph()
    assert nx.is_bipartite(graph) and nx.number_of_selfloops(graph) == 0
    assert all(d['bipartite'] in {'author', 'video'} for _, d in graph.nodes(data=True))
    assert all(graph.nodes[u]['bipartite'] != graph.nodes[v]['bipartite'] for u, v in graph.edges())
    return graph


def project(graph, kind):
    """Cuenta vecinos comunes distintos y conserva incluso nodos aislados."""
    result = nx.Graph()
    nodes = sorted(n for n, d in graph.nodes(data=True) if d['bipartite'] == kind)
    result.add_nodes_from((n, dict(graph.nodes[n])) for n in nodes)
    for other in sorted(set(graph) - set(nodes)):
        for u, v in combinations(sorted(graph[other]), 2):
            if result.has_edge(u, v):
                result[u][v]['weight'] += 1
            else:
                result.add_edge(u, v, weight=1)
    assert set(result) == set(nodes)
    assert nx.number_of_selfloops(result) == 0
    for u, v, d in result.edges(data=True):
        assert isinstance(d['weight'], int) and 0 < d['weight'] <= len(graph) - len(nodes)
        assert d['weight'] == len(set(graph[u]) & set(graph[v]))
    return result


def network_summary(graph):
    n, e = graph.number_of_nodes(), graph.number_of_edges()
    components = list(nx.connected_components(graph))
    weights = [d['weight'] for _, _, d in graph.edges(data=True)]
    mean_degree = 2 * e / n if n else 0
    assert np.isclose(mean_degree, np.mean([d for _, d in graph.degree()]) if n else 0)
    return dict(n_nodes=n, n_edges=e, n_isolates=nx.number_of_isolates(graph),
                density=nx.density(graph), mean_degree=mean_degree,
                min_weight=min(weights) if weights else float('nan'),
                mean_weight=float(np.mean(weights)) if weights else float('nan'),
                max_weight=max(weights) if weights else float('nan'),
                n_components=len(components), largest_component_size=max(map(len, components), default=0))


def export_projection(graph, name):
    nodes = pd.DataFrame([dict(node_id=n, **d) for n, d in graph.nodes(data=True)])
    edges = pd.DataFrame([dict(source=u, target=v, weight=d['weight'])
                          for u, v, d in graph.edges(data=True)], columns=['source', 'target', 'weight'])
    nodes.to_csv(NETWORK_DIR / f'{name}_nodes.csv', index=False)
    edges.to_csv(NETWORK_DIR / f'{name}_edges.csv', index=False)
    nx.write_graphml(graph, NETWORK_DIR / f'{name}.graphml')
    # Comprobar la reconstruccion desde CSV, incluidos los aislados y atributos.
    saved_nodes = pd.read_csv(NETWORK_DIR / f'{name}_nodes.csv', keep_default_na=False)
    saved_edges = pd.read_csv(NETWORK_DIR / f'{name}_edges.csv')
    rebuilt = nx.from_pandas_edgelist(saved_edges, 'source', 'target', 'weight')
    rebuilt.add_nodes_from(saved_nodes.node_id)
    assert set(rebuilt) == set(graph)
    assert {frozenset((u, v)): d['weight'] for u, v, d in rebuilt.edges(data=True)} == {
        frozenset((u, v)): d['weight'] for u, v, d in graph.edges(data=True)}


def plot_projection(graph, filename, title, communities=None):
    fig, ax = plt.subplots(figsize=(13, 9))
    pos = nx.spring_layout(graph, seed=RANDOM_SEED, weight='weight', iterations=150)
    is_video = all(d['bipartite'] == 'video' for _, d in graph.nodes(data=True))
    if is_video:
        # Layout circular completo: los aislados no comprimen la componente
        # conectada. Numeros y clave lateral evitan superponer titulos.
        pos = nx.circular_layout(graph)
    # Area proporcional a participacion con base visible para todos los nodos.
    sizes = [40 + 3 * d.get('n_comentarios', d.get('n_comentarios_totales', 0))
             for _, d in graph.nodes(data=True)]
    colors = [communities[n] for n in graph] if communities else '#5296bd'
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.5 if is_video else 0.06,
                           width=[0.6 * d['weight'] for _, _, d in graph.edges(data=True)])
    dots = nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=sizes, node_color=colors,
                                 cmap=plt.get_cmap('tab20') if communities else None)
    # Todos los videos; solo 5 autores con mayor grado (desempate por ID).
    labeled = list(graph) if is_video else sorted(graph, key=lambda n: (-graph.degree(n), n))[:5]
    labels = {n: str(i) for i, n in enumerate(graph, 1)} if is_video else {
        n: graph.nodes[n]['label'][:30] for n in labeled}
    nx.draw_networkx_labels(graph, pos, labels=labels,
                            font_size=7, ax=ax)
    if is_video:
        key = '\n'.join(f"{i:2}. {graph.nodes[n]['label'][:43]}" for i, n in enumerate(graph, 1))
        ax.text(1.03, 0.5, key, transform=ax.transAxes, va='center', fontsize=8, linespacing=1.6)
    if communities:
        from matplotlib.colors import BoundaryNorm
        ids = sorted(set(communities.values()))
        dots.set_cmap(plt.get_cmap('tab20', len(ids)))
        dots.set_norm(BoundaryNorm(np.arange(0.5, len(ids) + 1.5), len(ids)))
        fig.colorbar(dots, ax=ax, ticks=ids, label='Comunidad', shrink=0.7, location='left')
    ax.set_title(title + f'\nRed completa: {len(graph)} nodos, {graph.number_of_edges()} aristas')
    ax.text(0.01, 0.01, 'Area: comentarios observados; grosor: peso de la arista', transform=ax.transAxes, fontsize=8)
    ax.margins(0.18)
    ax.axis('off')
    fig.savefig(FIGURES_DIR / filename, bbox_inches='tight', dpi=150)
    plt.close(fig)


def main():
    bipartite = load_bipartite()
    comments = pd.read_csv(COMMENTS_CLEAN_CSV, keep_default_na=False)
    observed = {(r.author_channel_id, r.video_id) for r in comments.itertuples()}
    actual = {(u, v) if bipartite.nodes[u]['bipartite'] == 'author' else (v, u)
              for u, v in bipartite.edges()}
    assert observed == actual
    author_info = comments.drop_duplicates('author_channel_id').set_index('author_channel_id')
    rows = []
    for kind, title in [('author', 'Autores: co-participacion en videos'), ('video', 'Videos: audiencia compartida')]:
        graph = project(bipartite, kind)
        for n, d in graph.nodes(data=True):
            if kind == 'author':
                d.update(author_name=d['label'], author_handle=str(author_info.loc[n, 'author_handle']))
            else:
                d['title'] = d['label']
        name = f'{kind}_projection'
        export_projection(graph, name)
        rows.append(dict(network=name, **network_summary(graph)))
        plot_projection(graph, f'05_{name}.png', title)
    summary = pd.DataFrame(rows)
    summary.to_csv(NETWORK_DIR / 'projection_comparison.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
