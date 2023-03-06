# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 2022

@author: redha

"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import networkx as nx
import sys
sys.path.append('source')
from models import DQN_buffer_model, DQN_buffer_FP_model, DQN_routing_model

if __name__ == '__main__':
    """
    This script pre train the DQN agents to learn shortest path routing and save the weights
    """
    
    ### define the params
    topology = '5n'#'4n' #'5n'
    size_of_data_per_dst = 1000 
    topology_name = "abilene"
    buffer_max_length = 270
    ### load the topology graph
    G=nx.Graph()

    for i, element in enumerate(np.loadtxt(open(f"examples/{topology_name}/node_coordinates_{topology}.txt"))):
        G.add_node(i,pos=tuple(element))
    G = nx.from_numpy_matrix(np.loadtxt(open(f"examples/{topology_name}/adjacency_matrix_2_{topology}.txt")), create_using=G)
    ping_mat = np.loadtxt(open(f"scripts/ping_{topology}_mat.txt"))
    #remove_list = [node for node,degree in dict(G.degree()).items() if degree < 1]
    #G.remove_nodes_from(remove_list)

    print(G.nodes())
    print(G.edges())
    nx.draw_networkx(G, with_labels=True)
    
    ### loop for nodes
    models = []
    for node in G.nodes():
    # for node in [6, 10]:
        number_of_neighbors = len(list(G.neighbors(node)))
        x_all = []
        y_all = []
        for dst in G.nodes():
            if dst == node:
                continue
            ### generate the random data for each destination
            x_dst = np.concatenate((dst * np.ones(shape=(size_of_data_per_dst, 1), dtype=int),
                                np.random.randint(low=0, high=buffer_max_length, size=(size_of_data_per_dst, number_of_neighbors)),
                                np.random.randint(low=0, high=10000, size=(size_of_data_per_dst, number_of_neighbors)),
                                # np.random.uniform(low=0, high=1, size=(size_of_data_per_dst, 1)),
                                # np.random.randint(low=0, high=3000, size=(size_of_data_per_dst, 1))
                                ),
                                axis=1)
            y_dst = []
            for interface_id, neighbor in enumerate(list(G.neighbors(node))):
                #print(dst, node, interface_id)
                cost = len(nx.shortest_path(G, neighbor, dst)) -1
                cost = (ping_mat[node][neighbor]+ping_mat[neighbor][dst])*0.001
                print(dst, node, neighbor, cost)
                y_dst_neighbor = cost * np.ones((size_of_data_per_dst, 1), dtype=int)
                if len(y_dst) == 0: 
                    y_dst = y_dst_neighbor
                else:
                    y_dst = np.concatenate((y_dst, y_dst_neighbor), axis=1)
            ### group the data into one tensor
            
            if len(x_all) == 0: 
                x_all = x_dst.copy()
                y_all = y_dst.copy()
            else:
                x_all = np.concatenate((x_all, x_dst), axis=0)
                y_all = np.concatenate((y_all, y_dst), axis=0)
        print(x_all, y_all)   
        ### load the model
        model = DQN_buffer_model(observation_shape=(1+number_of_neighbors+number_of_neighbors, ),
                 num_actions=number_of_neighbors, 
                 num_nodes=G.number_of_nodes(), 
                 input_size_splits=[1,number_of_neighbors,number_of_neighbors])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                      loss=keras.losses.MeanSquaredError(),
                      metrics=[keras.metrics.MeanSquaredError()]
                      )
        model.fit(x_all, y_all, batch_size=128, epochs=100)
        ### saving the model    
        model.save(f"examples/{topology_name}/dqn_buffer_with_throughputs_sp_init_overlay_modified_{topology}/node{node}")
        print(f"examples/{topology_name}/dqn_buffer_with_throughputs_sp_init_overlay_modified_{topology}/node{node}")