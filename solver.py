import networkx as nx
import numpy as np
from networkx.algorithms.approximation import min_weighted_dominating_set
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys
import os

adds = 0
remove_v = 0
switch_e = 0

def generate_new_solution(G, T):
    """With half probability decide to add or remove a vertex. If add a vertex, choose a vertex at random and add one of its
    neighbors that is not already within the graph. If remove vertex, choose a leaf at random, remove from tree, but only do
    so if the resulting graph is still a dominating set. Return the resulting solution tree."""
    global adds, remove_v, switch_e
    roll = np.random.random()
    if (roll < 0.33):   # add a vertex
        tree_vertices = list(T.nodes)
        rand_vertex = np.random.choice(tree_vertices)
        rand_neighbors = list(G.adj[rand_vertex])
        rand_neighbors = [rn for rn in rand_neighbors if rn not in tree_vertices] # ensure candidate neighbors not already inside the tree
        if not rand_neighbors: # empty list
            return generate_new_solution(G, T)
        else:
            adds += 1
            chosen = np.random.choice(rand_neighbors)
            T.add_node(chosen)
            T.add_edge(chosen, rand_vertex, weight = G.get_edge_data(chosen, rand_vertex)['weight'])
            return T

    elif (roll < 0.67): # remove a leaf
        tree_vertices = list(T.nodes)
        T_leaves = [l for l in tree_vertices if T.degree(l) == 1]
        np.random.shuffle(T_leaves)
        for leaf in T_leaves:
            T_prime = T.copy()
            T_prime.remove_node(leaf)
            if is_valid_network(G,T_prime):
                remove_v += 1
                return T_prime
        return generate_new_solution(G,T)

    else: # switch an edge
        switch_e += 1
        tree_edges = list(T.edges)
        rand_index = np.random.choice(len(tree_edges))
        rand_edge = tree_edges[rand_index]
        T.remove_edge(rand_edge[0], rand_edge[1])
        cc = list(nx.connected_components(T))
        # randomly adding back in an edge to reconnect the two CCs
        for vertex_a in cc[0]:
            for vertex_b in cc[1]:
                if G.has_edge(vertex_a,vertex_b):
                    T.add_edge(vertex_a, vertex_b, weight = G.get_edge_data(vertex_a, vertex_b)['weight'])
                    return T

def simulated_annealing(G, T, steps = 10000):
    """Takes in a possible solution tree T, original graph G, and executes a simulated_annealing procedure.
    At the conclusion returns the final T. """
    temp = [steps-i for i in range(steps)]
    calculate_prob = lambda iteration, delta: np.exp(-1 * delta / temp[iteration])
    for i in range(steps):
        T_prime = generate_new_solution(G, T)
        delta = average_pairwise_distance(T_prime) - average_pairwise_distance(T)
        if delta < 0:
            T = T_prime
        elif np.random.random() < calculate_prob(i,delta):
            T = T_prime
    return T

def solve(G, target):
    """
    Pure simulated annealing. Start with an arbitrary MST though.
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """
    # target = 75
    output_tree = nx.minimum_spanning_tree(G, weight = 'weight')
    while True:
        output_tree =  simulated_annealing(G, output_tree, 100)
        output_cost = average_pairwise_distance(output_tree)
        print(output_cost)
        if (output_cost < target):
            return output_tree


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in
# Below for testing one at a time:

if __name__ == '__main__':
    assert len(sys.argv) == 3#2
    path = sys.argv[1]
    output_path = "outputs/" + path[7:]
    target = int(sys.argv[2])
    # print(output_path)
    G = read_input_file(path)
    T = solve(G, target)
    assert is_valid_network(G, T)
    print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    #print("Num adds " + str(adds))
    #print("Num leaf removals " + str(remove_v))
    #print("Num edge switches " + str(switch_e))
    write_output_file(T, output_path)
