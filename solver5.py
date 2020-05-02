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
        G.nodes[vertex]['avg_distances'] = total / (num_vertices_in_graph - 1)

    dominatingSet = nx.algorithms.approximation.dominating_set.min_weighted_dominating_set(G, weight="avg_distances")

    # Create the graph T and input the dominating set.
    T = nx.Graph()
    T.add_nodes_from(dominatingSet)

    # When the graph is fully dominated by one node, the cost is zero.
    if len(dominatingSet) == 1:
        return T

    # Create a list of tuples, where each tuple holds (cost, dominating_set_tree). Perform for every possible dominating vertex starting point.
    candidates_list = []

    for starting_point in dominatingSet:
        T_candidate = T.copy()
        distances_from_start_point = []
        # Determine which vertices are the closest and furthest away from the starting vertex
        # Will do this by making a python list of tuples (dominating vertex, distance) and sorting by distance
        for dom_v in dominatingSet:
            if dom_v != starting_point:
                shortest_distance = apsp[starting_point][0][dom_v]
                distances_from_start_point.append((dom_v, shortest_distance))
        # Sort the dominating vertices by increasing order away from random_start
        distances_from_start_point.sort(key = lambda t: t[1])
        # Connect the vertices in order of distance away from start vertex.
        ordered_vertices_to_connect = [t[0] for t in distances_from_start_point]
        # In each iteration of the following for loop, we connect node to node2.
        node2 = starting_point
        for node in ordered_vertices_to_connect:
            connected_nodes = nx.descendants(T_candidate, node2)
            if node2 is not None: # and node2 not in connected_nodes:
                # We want to connect node to node2 in the cheapest way possible. Thus, we search from all nodes connected to node2 and choose shortest path.
                shortest_path = apsp[node2][1][node]
                path_cost = apsp[node2][0][node]
                # Now, evaluate if there is a shorter, cheaper path from a different vertex.
                for connected_node in connected_nodes:
                    if apsp[connected_node][0][node] < path_cost:
                        shortest_path = apsp[connected_node][1][node]
                        path_cost = apsp[connected_node][0][node]
                for vertex in shortest_path:   # Add all necessary vertices
                    if vertex not in dominatingSet:
                        T_candidate.add_node(vertex)
                for i in range(len(shortest_path) - 1):   # Add all necessary edges
                    origin = shortest_path[i]
                    terminus = shortest_path[i+1]
                    T_candidate.add_edge(origin, terminus, weight=G.get_edge_data(origin, terminus)['weight'])
            node2 = node

        candidates_list.append((average_pairwise_distance(T_candidate), T_candidate))

    ### Part 2: Identifying best candidate and endgame optimization.
    candidates_list.sort(key = lambda t: t[0])
    current_average = candidates_list[0][0]
    T = candidates_list[0][1]
    last_average = 4000

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
