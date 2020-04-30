import networkx as nx
from networkx.algorithms.approximation import min_weighted_dominating_set
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys


def solve(G):
    """
    Directly calls MST algorithm to link together the nodes of the dominatingSet. Only includes edges that link together two dominating set vertices.
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """
    temp = nx.Graph()
    dominatingSet = min_weighted_dominating_set(G, weight="weight")
    temp.add_nodes_from(dominatingSet)

    for node in dominatingSet:
        for node2 in dominatingSet:
            if G.has_edge(node, node2):
                temp.add_edge(node, node2)
                temp[node][node2]['weight'] = G.get_edge_data(node, node2)['weight']

    # Get MST of dominating set
    edges = list(nx.minimum_spanning_edges(temp, algorithm='kruskal', weight='weight', keys=True, data=True, ignore_nan=False))
    T = nx.Graph()
    T.add_nodes_from(dominatingSet)
    T.add_edges_from(edges)

    return T

# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G = read_input_file(path)
    T = solve(G)
    assert is_valid_network(G, T)
    print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    write_output_file(T, 'out/test.out')
