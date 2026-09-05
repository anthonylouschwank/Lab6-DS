"""Ejercicio 6: topologia, componentes y periferia observada.

Densidad general = 2E/(N(N-1)); se agrega E/(autores*videos) por separado
para la bipartita. Transitividad = 3*triangulos/triadas y clustering medio
= media de coeficientes locales, incluyendo ceros, SIN pesos; solamente
para las proyecciones. No son medidas de intensidad ni se aplican a la
bipartita, cuyos triangulos estan excluidos por construccion.
Fuentes: https://networkx.org/documentation/stable/reference/algorithms/
 generated/networkx.algorithms.cluster.transitivity.html
 generated/networkx.algorithms.cluster.average_clustering.html
Una componente pequena tiene <= 3 nodos; grado bajo/pocos autores <= 2.
Son umbrales descriptivos, no pruebas estadisticas. Aislamiento observado
no implica ausencia real de conexiones en YouTube.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from config import NETWORK_DIR, FIGURES_DIR
from p5_network_projections import load_bipartite, network_summary


def component_tables(graph):
    components = sorted(nx.connected_components(graph), key=lambda c: (-len(c), min(c)))
    rows, members = [], []
    for cid, nodes in enumerate(components, 1):
        rows.append(dict(component_id=cid, n_nodes=len(nodes), n_edges=graph.subgraph(nodes).number_of_edges(),
                         is_main=cid == 1, is_small=len(nodes) <= 3))
        for n in sorted(nodes):
            members.append(dict(node_id=n, label=graph.nodes[n]['label'], component_id=cid,
                                component_size=len(nodes), is_main=cid == 1, is_small=len(nodes) <= 3,
                                degree=graph.degree(n), is_isolate=graph.degree(n) == 0))
    assert sum(len(c) for c in components) == len(graph)
    mapping = {r['node_id']: r['component_id'] for r in members}
    assert len(mapping) == len(graph) and all(mapping[u] == mapping[v] for u, v in graph.edges())
    return pd.DataFrame(rows), pd.DataFrame(members)


def topology_metrics(graph, name):
    result = dict(network=name, **network_summary(graph))
    degrees = np.array([d for _, d in graph.degree()])
    assert len(degrees) > 0, 'Se requiere al menos un nodo observado'
    result.update(largest_component_share=result['largest_component_size'] / len(graph),
                  median_degree=float(np.median(degrees)), max_degree=int(degrees.max()),
                  share_degree_0=float(np.mean(degrees == 0)), share_degree_1=float(np.mean(degrees == 1)))
    for p in [25, 75, 90, 95]:
        result[f'degree_p{p}'] = float(np.percentile(degrees, p))
    is_bipartite = name == 'bipartite'
    result['transitivity'] = float('nan') if is_bipartite else nx.transitivity(graph)
    result['average_clustering'] = float('nan') if is_bipartite else nx.average_clustering(graph, weight=None, count_zeros=True)
    result['clustering_scope'] = 'no_aplica_triangular_bipartita' if is_bipartite else 'sin_pesos_incluye_ceros'
    n_authors = sum(d['bipartite'] == 'author' for _, d in graph.nodes(data=True))
    result['bipartite_density'] = graph.number_of_edges() / (n_authors * (len(graph) - n_authors)) if is_bipartite else float('nan')
    return result


def main():
    graphs = {'bipartite': load_bipartite(),
              'author_projection': nx.read_graphml(NETWORK_DIR / 'author_projection.graphml'),
              'video_projection': nx.read_graphml(NETWORK_DIR / 'video_projection.graphml')}
    metrics, top_nodes = [], []
    fig_components, axes = plt.subplots(3, 1, figsize=(10, 11))
    for ax, (name, graph) in zip(axes, graphs.items()):
        assert not graph.is_directed() and nx.number_of_selfloops(graph) == 0
        metrics.append(topology_metrics(graph, name))
        components, members = component_tables(graph)
        components.to_csv(NETWORK_DIR / f'06_{name}_components.csv', index=False)
        members.to_csv(NETWORK_DIR / f'06_{name}_component_members.csv', index=False)
        ranked = members.sort_values(['degree', 'node_id'], ascending=[False, True]).head(10).copy()
        ranked.insert(0, 'network', name)
        top_nodes.append(ranked)
        degrees = pd.Series(dict(graph.degree()))
        freq = degrees.value_counts().sort_index().rename_axis('degree').reset_index(name='n_nodes')
        freq['share_nodes'] = freq.n_nodes / len(graph)
        assert freq.n_nodes.sum() == len(graph)
        assert (freq.degree * freq.n_nodes).sum() == 2 * graph.number_of_edges()
        short = name.replace('_projection', '')
        freq.to_csv(NETWORK_DIR / f'06_{short}_degree_distribution.csv', index=False)
        fig, degree_ax = plt.subplots(figsize=(10, 5))
        degree_ax.bar(freq.degree, freq.n_nodes)
        degree_ax.set(xlabel='Grado (sin pesos)', ylabel='Numero de nodos', title=f'Distribucion de grados: {name}')
        fig.savefig(FIGURES_DIR / f'06_{short}_degree_distribution.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        ax.bar(components.component_id.astype(str), components.n_nodes)
        ax.set(xlabel='Componente (orden descendente de tamano)', ylabel='Numero de nodos', title=name)
        if name != 'bipartite':
            attrs = pd.DataFrame([dict(node_id=n, **d) for n, d in graph.nodes(data=True)])
            peripheral = members.merge(attrs, on='node_id', validate='one_to_one', suffixes=('', '_attribute'))
            peripheral['low_degree'] = peripheral.degree <= 2
            if name == 'author_projection':
                peripheral['single_video_author'] = peripheral.n_videos_distintos == 1
            else:
                peripheral['few_observed_authors'] = peripheral.n_autores_unicos <= 2
            peripheral['scope'] = 'solo_muestra_observada'
            peripheral.to_csv(NETWORK_DIR / f'06_{short}_periphery.csv', index=False)
    fig_components.tight_layout()
    fig_components.savefig(FIGURES_DIR / '06_component_sizes.png', bbox_inches='tight', dpi=150)
    plt.close(fig_components)
    summary = pd.DataFrame(metrics)
    summary.to_csv(NETWORK_DIR / '06_network_metrics.csv', index=False)
    pd.concat(top_nodes, ignore_index=True).to_csv(NETWORK_DIR / '06_top_degree_nodes.csv', index=False)
    print(summary[['network', 'n_nodes', 'n_edges', 'n_components', 'n_isolates', 'largest_component_share']].to_string(index=False))
    print('Periferia: grado <= 2, componentes pequenas <= 3 nodos; aislamiento solo observado.')


if __name__ == '__main__':
    main()
