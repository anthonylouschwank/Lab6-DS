"""Controles con redes pequenas cuyo resultado puede verificarse a mano.
Ejecutar: python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p5_network_projections import project
from p6_topology_fragmentation import component_tables, topology_metrics
from p7_communities import detect_communities


class NetworkAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_nodes_from((n, {'bipartite': 'author', 'label': n}) for n in ['a', 'b', 'c'])
        self.graph.add_nodes_from((n, {'bipartite': 'video', 'label': n}) for n in ['x', 'y', 'z'])
        self.graph.add_weighted_edges_from([('a', 'x', 8), ('a', 'y', 1), ('b', 'x', 2), ('b', 'y', 5), ('c', 'z', 3)])

    def test_distinct_neighbors_not_comment_counts(self):
        authors = project(self.graph, 'author')
        videos = project(self.graph, 'video')
        self.assertEqual(authors['a']['b']['weight'], 2)
        self.assertEqual(videos['x']['y']['weight'], 2)
        self.assertEqual(set(nx.isolates(authors)), {'c'})
        self.assertEqual(set(nx.isolates(videos)), {'z'})
        self.assertEqual(authors.number_of_edges(), 1)
        self.assertEqual(videos.number_of_edges(), 1)

    def test_components_and_degree_identity(self):
        graph = project(self.graph, 'video')
        sizes, members = component_tables(graph)
        self.assertEqual(sizes.n_nodes.tolist(), [2, 1])
        self.assertEqual(members.set_index('node_id').loc['z', 'component_size'], 1)
        metrics = topology_metrics(graph, 'video_projection')
        self.assertAlmostEqual(metrics['mean_degree'], 2 / 3)
        self.assertAlmostEqual(metrics['share_degree_0'], 1 / 3)
        self.assertEqual(metrics['transitivity'], 0)
        triangle = nx.complete_graph(3)
        nx.set_edge_attributes(triangle, 1, 'weight')
        nx.set_node_attributes(triangle, 'video', 'bipartite')
        self.assertEqual(topology_metrics(triangle, 'video_projection')['transitivity'], 1)

    def test_communities_keep_isolates_and_are_repeatable(self):
        graph = project(self.graph, 'video')
        mapping, modularity = detect_communities(graph)
        self.assertEqual(mapping['x'], mapping['y'])
        self.assertNotEqual(mapping['x'], mapping['z'])
        self.assertEqual(detect_communities(graph), (mapping, modularity))
        self.assertEqual(len(detect_communities(nx.empty_graph(3))[0]), 3)
        self.assertEqual(detect_communities(nx.Graph())[0], {})

    def test_louvain_uses_weighted_structure(self):
        graph = nx.cycle_graph(4)
        nx.set_edge_attributes(graph, 1, 'weight')
        graph[0][1]['weight'] = graph[2][3]['weight'] = 100
        mapping, q = detect_communities(graph)
        self.assertEqual(mapping[0], mapping[1])
        self.assertEqual(mapping[2], mapping[3])
        self.assertNotEqual(mapping[0], mapping[2])
        self.assertGreater(q, 0.4)


if __name__ == '__main__':
    unittest.main()
