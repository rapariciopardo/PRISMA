import networkx as nx
import numpy as np
import os


## define the parameters
link_cap = 500000 #bps
hop_delay = 0.001 #s
pkt_size = 542 #bytes
nb_nodes = 5
root_path = "/home/redha/PRISMA/prisma/examples/abilene"
exps_folder_name = f"dqn_buffer_{nb_nodes}n_moving_avg_variation"
## get the buffers raw data
for session_name in os.listdir(f"{root_path}/{exps_folder_name}/"):
# session_name = "dqn_buffer_5"
    if session_name in ("saved_models", "opt", "sp"):
        continue
    raw_data_path = f"{root_path}/{exps_folder_name}/{session_name}/groundTruth.txt"
    raw_data = np.loadtxt(raw_data_path)

    ## load overlay topology
    overlay_topology_path = f"{root_path}/adjacency_matrix_2_{nb_nodes}n.txt"
    G_overlay = nx.from_numpy_array(np.loadtxt(overlay_topology_path), create_using=nx.DiGraph())

    ## load underlay topology
    underlay_topology_path = f"{root_path}/adjacency_matrix.txt"
    G_underlay = nx.from_numpy_array(np.loadtxt(underlay_topology_path), create_using=nx.DiGraph())

    ## load correspondance vector
    correspondance_vector_path = f"/home/redha/PRISMA/prisma/mapOverlay_{nb_nodes}n.txt"
    correspondance_vector = np.loadtxt(correspondance_vector_path).tolist()

    delays_data = np.zeros((raw_data.shape[0], G_overlay.edges.__len__()+1))
    delays_data[:, 0] = raw_data[:, 0]
    ## for each virtual link, list the underlay edges
    for idx_edge, edge in enumerate(G_overlay.edges):
        if idx_edge == 0:
            actual_node = edge[0]
            interface_idx = -1
        ## get src and dst of the link
        virtual_node_src = edge[0]
        virtual_node_dst = edge[1]
        interface_idx +=1
        if virtual_node_src != actual_node:
            interface_idx = 0
            actual_node = virtual_node_src 
        
        ## transform to underlay
        underlay_node_src = correspondance_vector.index(virtual_node_src)
        underlay_node_dst = correspondance_vector.index(virtual_node_dst)
        
        ## get the underlay path
        underlay_path = nx.shortest_path(G_underlay, underlay_node_src, underlay_node_dst)
        
        ## sum the values of the 
        for idx in range(len(underlay_path)-1):
            index_in_raw_data = list(G_underlay.edges).index((underlay_path[idx], underlay_path[idx+1]))
            delays_data[:, idx_edge+1] += ((raw_data[:, index_in_raw_data+1] + pkt_size) *8 /link_cap)+hop_delay
        np.savetxt(f"{root_path}/{exps_folder_name}/{session_name}/ground_truth_delay_{virtual_node_src}_{interface_idx}.txt", delays_data[:, [0, idx_edge+1]])
    # print(f"{root_path}/{exps_folder_name}/{session_name}/ground_truth_delay_{virtual_node_src}_{interface_idx}.txt")
# print(delays_data)