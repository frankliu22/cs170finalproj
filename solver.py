import networkx as nx
from networkx.algorithms.approximation import min_weighted_dominating_set
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys
import os



def solve(G):
    """
    Similar to Algo 2 submission except more expansive in that it starts at every possible vertex in dominating set and tries
    to find connections from there.
    Args:
        G: networkx.Graph
    Returns:
        T: networkx.Graph
    """
    # The variable defined below, apsp, stands for all pairs shortest paths from calling NetworkX built-in Dijkstra's algorithm.
    apsp = dict(nx.algorithms.shortest_paths.weighted.all_pairs_dijkstra(G, weight = "weight"))

    num_vertices_in_graph = len(list(G.nodes))
    if num_vertices_in_graph == 1:
        return G

    for vertex in list(G.nodes):
        total = 0
        for vertex2 in list(G.nodes):
            total += apsp[vertex][0][vertex2]
        avg = total / (num_vertices_in_graph - 1)
        G.nodes[vertex]['avg_distances'] = avg

    dominatingSet = nx.algorithms.approximation.dominating_set.min_weighted_dominating_set(G, weight="avg_distances")

    # Create the graph T and input the dominating set.
    T = nx.Graph()
    T.add_nodes_from(dominatingSet)

    # When the graph is fully dominated by one node, the cost is zero.
    if len(dominatingSet) == 1:
        return T

    # Identify lightest vertex from which to start Prim's.
    lightest_vertex = None
    shortest_avg_distance = 5000
    for vertex in dominatingSet:
        avg_dist = G.nodes[vertex]['avg_distances']
        if (avg_dist < shortest_avg_distance):
            lightest_vertex = vertex
            shortest_avg_distance = avg_dist

    connected = {lightest_vertex}   # the lightest_vertex (closest in proximity to all others) will be our starting point
    still_to_connect = dominatingSet
    still_to_connect.remove(lightest_vertex)

    while len(still_to_connect) != 0:   # while there are still vertices to connect
        shortest_connecting_path = None
        shortest_connecting_distance = 5000
        for c in connected:
            for s in still_to_connect:
                candidate_dist = apsp[c][0][s]
                if (candidate_dist < shortest_connecting_distance):
                    shortest_connecting_distance = candidate_dist
                    shortest_connecting_path = apsp[c][1][s]
        for vertex in shortest_connecting_path:   # add all the vertices on this shortest path
            if vertex not in dominatingSet:
                T.add_node(vertex)
        for i in range(len(shortest_connecting_path) - 1):   # Add all necessary edges along shortest path
            origin = shortest_connecting_path[i]
            terminus = shortest_connecting_path[i+1]
            T.add_edge(origin, terminus, weight=G.get_edge_data(origin, terminus)['weight'])
        s = shortest_connecting_path[-1]
        connected.add(s)
        still_to_connect.remove(s)

    ### Part 2: Endgame optimization.
    current_average = average_pairwise_distance(T)

    T_vertices = list(T.nodes)
    G_vertices = list(G.nodes)
    corona = [gv for gv in G_vertices if gv not in T_vertices]

    for cv in corona:
        # Look at all edges leading out of the coronavertex and pick the cheapest one to consider adding.
        T_vertices = list(T.nodes)
        cv_neighbors = [n for n in list(G.adj[cv]) if n in T_vertices]
        cv_edges = [(cn, G.get_edge_data(cv, cn)['weight']) for cn in cv_neighbors]
        cheapest_edge = min(cv_edges, key = lambda t: t[1])  # can only make one connection from cv into the tree
        connection_point = cheapest_edge[0]   # identify connection point within tree
        connection_weight = cheapest_edge[1]
        if connection_weight < current_average:
            T.add_node(cv)
            T.add_edge(cv,connection_point, weight = connection_weight)
            new_average = average_pairwise_distance(T)
            if new_average >= current_average:   # uh oh, got heavier
                T.remove_node(cv)
            else:
                current_average = new_average

    return T


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in
# Below for testing one at a time:

if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    output_path = "outputs/" + path[7:]
    # print(output_path)
    G = read_input_file(path)
    T = solve(G)
    assert is_valid_network(G, T)
    print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    #print("Num adds " + str(adds))
    #print("Num leaf removals " + str(remove_v))
    #print("Num edge switches " + str(switch_e))
    write_output_file(T, output_path)
