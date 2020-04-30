import networkx as nx
from networkx.algorithms.approximation import min_weighted_dominating_set
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys
import os

def solver_alternate(G):
    """Basically consists of the old algorithmic approach using Nick's method of Prim's algorithm. Also takes in G and returns T."""
    dominatingSet = nx.algorithms.approximation.dominating_set.min_weighted_dominating_set(G, weight="weight")

    # Create the graph T and input the dominating set.
    T = nx.Graph()
    T.add_nodes_from(dominatingSet)

    # When the graph is fully dominated by one node, the cost is zero.
    if len(dominatingSet) == 1:
        return T

    # The variable defined below, apsp, stands for all pairs shortest paths from calling NetworkX built-in Dijkstra's algorithm.
    apsp = dict(nx.algorithms.shortest_paths.weighted.all_pairs_dijkstra(G, weight = "weight"))

    # In each iteration of the following for loop, we connect node to node2.
    node2 = None
    for node in dominatingSet:

        connected_nodes = nx.dfs_preorder_nodes(T, source=node2)

        print("Connecting ", node, "to", node2)
        print(node2, "descendants: ", [n for n in connected_nodes])

        if node2 is not None and node2 not in list(connected_nodes):

            # We want to connect node to node2 in the cheapest way possible.
            # Thus, we search from all nodes connected to node2.
            # By default, we choose the shortest path from node to node2.
            shortest_path = apsp[node2][1][node]
            path_cost = apsp[node2][0][node]

            # Now, evaluate if there is a shorter, cheaper path from a different vertex.
            for connected_node in list(nx.dfs_preorder_nodes(T, source=node2)):
                print("    Cheapest path from " + str(connected_node) + ": " + str(apsp[connected_node][0][node]))
                if apsp[connected_node][0][node] < path_cost:
                    shortest_path = apsp[connected_node][1][node]
                    path_cost = apsp[connected_node][0][node]
                    print("Updating best path to " + str(shortest_path) + ", cost " + str(path_cost))

            print("Found path via " + str(shortest_path) + " of cost " + str(path_cost))

            # Add all necessary vertices.
            for vertex in shortest_path:
                if vertex not in dominatingSet:
                    T.add_node(vertex)

            # Add all necessary edges.
            for i in range(len(shortest_path) - 1):
                origin = shortest_path[i]
                terminus = shortest_path[i+1]
                T.add_edge(origin, terminus, weight=G.get_edge_data(origin, terminus)['weight'])
                #T[origin][terminus]['weight'] = G.get_edge_data(origin, terminus)['weight']

        node2 = node
        print()

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


def solve(G):
    """
    Assign each vertex a avg_weight attribute equal to the average weight of its neighboring edges. Then call the standard dominatingSet
    function, connect the vertices of the dominating set using Nick's Prim's algo, and call extra-vertex-optimization in the end.
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """
    # The variable defined below, apsp, stands for all pairs shortest paths from calling NetworkX built-in Dijkstra's algorithm.
    apsp = dict(nx.algorithms.shortest_paths.weighted.all_pairs_dijkstra(G, weight = "weight"))

    num_vertices_in_graph = len(list(G.nodes))

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

    # We now want to determine an ordering of distances within the vertices of the dominating set.
    random_start = None
    for v in G.nodes:
        if v in dominatingSet:
            random_start = v
            break

    # Determine which vertices are the closest and furthest away from the starting vertex
    # Will do this by making a python list of tuples (dominating vertex, distance) and sorting by distance
    distances_from_random_start = []
    for dom_v in dominatingSet:
        if dom_v != random_start:
            shortest_distance = apsp[random_start][0][dom_v]
            distances_from_random_start.append((dom_v, shortest_distance))

    # Sort the dominating vertices by increasing order away from random_start
    distances_from_random_start.sort(key = lambda t: t[1])

    # Redefine the dominating set variable to hold the dominating variables in order of distance!
    dominatingSet = [random_start] + [t[0] for t in distances_from_random_start]

    # In each iteration of the following for loop, we connect node to node2.
    node2 = None
    for node in dominatingSet:

        connected_nodes = nx.dfs_preorder_nodes(T, source=node2)

        print("Connecting ", node, "to", node2)
        print(node2, "descendants: ", [n for n in connected_nodes])

        if node2 is not None and node2 not in list(connected_nodes):

            # We want to connect node to node2 in the cheapest way possible.
            # Thus, we search from all nodes connected to node2.
            # By default, we choose the shortest path from node to node2.
            shortest_path = apsp[node2][1][node]
            path_cost = apsp[node2][0][node]

            # Now, evaluate if there is a shorter, cheaper path from a different vertex.
            for connected_node in list(nx.dfs_preorder_nodes(T, source=node2)):
                print("    Cheapest path from " + str(connected_node) + ": " + str(apsp[connected_node][0][node]))
                if apsp[connected_node][0][node] < path_cost:
                    shortest_path = apsp[connected_node][1][node]
                    path_cost = apsp[connected_node][0][node]
                    print("Updating best path to " + str(shortest_path) + ", cost " + str(path_cost))

            print("Found path via " + str(shortest_path) + " of cost " + str(path_cost))

            # Add all necessary vertices.
            for vertex in shortest_path:
                if vertex not in dominatingSet:
                    T.add_node(vertex)

            # Add all necessary edges.
            for i in range(len(shortest_path) - 1):
                origin = shortest_path[i]
                terminus = shortest_path[i+1]
                T.add_edge(origin, terminus, weight=G.get_edge_data(origin, terminus)['weight'])
                #T[origin][terminus]['weight'] = G.get_edge_data(origin, terminus)['weight']

        node2 = node
        print()


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
                        #print("Adding an edge between", node, "and", node2, "yields average", new_average)



    #print("Dominating vertices:", [node for node in T])

    T_alternate = solver_alternate(G)
    alternate_average = average_pairwise_distance(T_alternate)
    if (current_average < alternate_average):
        return T
    else:
        return T_alternate

# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in
# Below for testing one at a time:
"""
if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G = read_input_file(path)
    T = solve(G)
    assert is_valid_network(G, T)
    print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    write_output_file(T, 'outputs/test.out')
"""
