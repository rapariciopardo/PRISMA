import networkx as nx
import numpy as np
import os

## load overlay topology
overlay_topology_path = "/home/tiago/Documents/PRISMA/prisma/examples/abilene/adjacency_matrix_2_5n.txt"
G_overlay = nx.from_numpy_array(np.loadtxt(overlay_topology_path), create_using=nx.DiGraph())

## load underlay topology
underlay_topology_path = "/home/tiago/Documents/PRISMA/prisma/examples/abilene/adjacency_matrix.txt"
G_underlay = nx.from_numpy_array(np.loadtxt(underlay_topology_path), create_using=nx.DiGraph())

## load correspondance vector
correspondance_vector_path = "/home/redha/PRISMA/prisma/mapOverlay_5n.txt"
correspondance_vector = np.loadtxt(correspondance_vector_path).tolist()

influence_matrix = np.zeros((len(list(G_overlay.nodes)), len(list(G_overlay.nodes))))
nodes_used_links = [set() for _ in G_overlay.nodes]
## for each virtual link, list the underlay edges
for idx_edge, edge in enumerate(G_overlay.edges):
    
    ## get src and dst of the link
    virtual_node_src = edge[0]
    virtual_node_dst = edge[1]
    
    ## transform to underlay
    underlay_node_src = correspondance_vector.index(virtual_node_src)
    underlay_node_dst = correspondance_vector.index(virtual_node_dst)
    
    ## get the underlay path
    underlay_path = nx.shortest_path(G_underlay, underlay_node_src, underlay_node_dst)
    
    ## sum the values of the 
    for idx in range(len(underlay_path)-1):
        nodes_used_links[virtual_node_src].add((underlay_path[idx], underlay_path[idx+1]))

for ni in G_overlay.nodes:
    for nj in G_overlay.nodes:
        if ni == nj:
            continue
        influence_matrix[ni][nj] = len(nodes_used_links[ni].intersection(nodes_used_links[nj]))
    
print()