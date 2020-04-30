import networkx as nx
from networkx.algorithms.approximation import min_weighted_dominating_set
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys


def solve(G):
    """
    Idea for solve method below: is to basically find shortest paths (via Dijkstras) between each pair of vertices within the dominating set.
    Put all the vertices within the dominating set into a new graph G_prime, and add edges between each pair of vertices. Introduce new vertices
    into the graph as necessary to ensure connectivity. In the end, return the MST of G_prime. Also calls extra-vertex-optimization in the end.

    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """

    # TODO: your code here!
    dominatingSet = min_weighted_dominating_set(G, weight = "weight")

    # The variable defined below, apsp, stands for all pairs shortest paths from calling NetworkX built-in Dijkstra's algorithm.
    apsp = dict(nx.algorithms.shortest_paths.weighted.all_pairs_dijkstra(G, weight = "weight"))

    # G_prime, the new graph below, shall consist of all vertices within the dominating set along with their shortest path edge weights in
    # between, bringing in new vertices as necessary.
    G_prime = nx.Graph()
    G.add_nodes_from(dominatingSet)

    # Vertices to add contains new vertices which must be added into graph G prime in order to ensure connectivity of nodes from min
    # dominating set.
    extra_vertices = set()

    for node in dominatingSet:
        for node2 in dominatingSet:
            shortest_path = apsp[node][1][node2]
            # First, identify new vertices to be thrown into G prime.
            for vertex in shortest_path:
                if vertex not in dominatingSet:
                    G_prime.add_node(vertex)   # I do believe from my Internet search that duplicate nodes has no effect
                    extra_vertices.add(vertex)   # Keep track of the list of all vertices within the dominating set
            # Next, identify new edges to be thrown into G prime. Adding edges more than once has no effect.
            for i in range(len(shortest_path) - 1):
                origin_vertex = shortest_path[i]
                terminus_vertex = shortest_path[i+1]
                w = G.get_edge_data(origin_vertex, terminus_vertex)['weight']
                G_prime.add_edge(origin_vertex, terminus_vertex, weight = w)

    final_edges = list(nx.minimum_spanning_edges(G_prime, algorithm='kruskal', weight='weight', keys=True, data=True, ignore_nan=False))

    T = nx.Graph()
    T.add_nodes_from(dominatingSet)
    T.add_nodes_from(extra_vertices)
    T.add_edges_from(final_edges)

    current_average = average_pairwise_distance(T)
    last_average = 4000
    print(current_average)

    # Until adding more edges doesn't improve the average pairwise cost
    while current_average < last_average:
        last_average = current_average
        # For every node in T
        for node in nx.dfs_preorder_nodes(T, source=list(T.nodes)[0]):
            neighbors = nx.algorithms.traversal.breadth_first_search.bfs_tree(G, node, reverse=False, depth_limit=1)
            # Get one of its neighbors NOT in T
            for node2 in neighbors:
                # and add the edge between that vertex and its neighbor
                # if it decreases the average pairwise cost.
                if node2 not in T and G.get_edge_data(node, node2)\
                and G[node][node2]['weight'] < current_average:
                    T.add_node(node2)
                    T.add_edge(node, node2, weight=G.get_edge_data(node, node2)['weight'])
                    new_average = average_pairwise_distance(T)
                    if new_average > current_average:
                        T.remove_node(node2)
                        #T.remove_edge(node, node2)
                    else:
                        current_average = new_average
                        print("Adding an edge between", node, "and", node2, "yields average", new_average)



    print("Dominating vertices:", [node for node in T])

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
