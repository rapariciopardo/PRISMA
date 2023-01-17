import networkx as nx
import numpy as np
import os

## define the parameters
link_cap = 500000 #bps
hop_delay = 0.001 #s
pkt_size = 542 #bytes

## get the buffers raw data
raw_data_path = "/home/tiago/Documents/PRISMA/prisma/examples/abilene/4nodes/test_log_4n/groundTruth.txt"
raw_data = np.loadtxt(raw_data_path)

## load overlay topology
overlay_topology_path = "/home/tiago/Documents/PRISMA/prisma/examples/abilene/adjacency_matrix_2_4n.txt"
G_overlay = nx.from_numpy_array(np.loadtxt(overlay_topology_path), create_using=nx.DiGraph())

## load underlay topology
underlay_topology_path = "/home/tiago/Documents/PRISMA/prisma/examples/abilene/adjacency_matrix.txt"
G_underlay = nx.from_numpy_array(np.loadtxt(underlay_topology_path), create_using=nx.DiGraph())

## load correspondance vector
correspondance_vector_path = "/home/redha/PRISMA/prisma/mapOverlay_4n.txt"
correspondance_vector = np.loadtxt(correspondance_vector_path).tolist()

delays_data = np.zeros((raw_data.shape[0], G_overlay.edges.__len__()+1))
delays_data[:, 0] = raw_data[:, 0]
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
        index_in_raw_data = list(G_underlay.edges).index((underlay_path[idx], underlay_path[idx+1]))
        delays_data[:, idx_edge+1] += ((raw_data[:, index_in_raw_data+1] + pkt_size) *8 /link_cap)+hop_delay

print(delays_data)