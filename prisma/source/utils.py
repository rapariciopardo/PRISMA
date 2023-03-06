# -*- coding: utf-8 -*-
from networkx.algorithms.clique import number_of_cliques
import tensorflow as tf
import os, multiprocessing, shutil
import numpy as np
from source.models import SplitLayer
import math

__author__ = "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__license__ = "GPL"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"


def save_model(actors, overlay_nodes, path, t, num_episodes, root="saved_models/"):
    """
    Save the DQN model for each node into a folder.

    Parameters
    ----------
    actors : list
        list of DQN models (one for each network node).
    path : str
        name of the folder where to store the models.
    t : int
        number of passed train iterations
    num_episodes : int
        number of passed episodes
    
    Returns
    -------
    None.

    """
    # if root not in os.listdir():
    #     os.mkdir(root)
    # if(path in os.listdir(root)):
    #     shutil.rmtree(root + path)
    if not(path in os.listdir(root)):
        os.mkdir(root + path)
    path = path.rstrip('/') + '/'
    folder_name = root + path + f"iteration{t}_episode{num_episodes}"
    for i in overlay_nodes:
        actors[i].q_network.save(f"{folder_name}/node{i}")


def load_model(path, node_index=-1):
    """
    Loads the list of agents from a directory

    Parameters
    ----------
    path : str
        The folder containing the saved models.

    Returns
    -------
    a list of Q functions for each agent. Return None if an error occurs

    """
    folders = os.listdir(path)
    q_functions = [1]*len(folders)
    for item in os.listdir(path):
        index = int(item.split("_")[-1][4:])
        print(index)
        if node_index >= 0 and node_index != index:
            continue
        try :
            q_functions[index] = tf.keras.models.load_model(path + "/" + item, custom_objects={"K":tf.keras.backend , "layers":tf.keras.layers, "SplitLayer":SplitLayer, "tf":tf}, compile=False)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
    return q_functions


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)



def convert_data_rate_to_bps(data_rate):
    """
    Convert the int data rate into text

    Args:
        data_rate (int): data rate in bps 
    """
    if data_rate == 0:
        return "0bps"
    size_name = ("bps", "Kbps", "Mbps", "Gbps", "Tbps")
    i = int(math.floor(math.log(data_rate, 1000)))
    p = math.pow(1000, i)
    s = round(data_rate / p, 2)
    return "%s%s" % (s, size_name[i])

def convert_bps_to_data_rate(bps):
    """
    Convert the text data rate into int

    Args:
        bps (str): data rate in text
    """

    unit = bps.rstrip("bps")
    size_name = ("K", "M", "G", "T")
    if unit[-1] in size_name:
        i = size_name.index(unit[-1])
        p = math.pow(1000, i+1)
        data_rate = float(unit[:-1])
    else:
        p = 1
        data_rate = float(unit)

    return data_rate*p

def optimal_routing_decision(graph, routing_mat, rejected_mat, actual_node, src_node, dst_node, tag):
    """Compute the action based on the optimal solution

    Args:
        graph (nx.graph): network graph
        routing_mat (np.array): optimal routing matrix
        lost_mat (np.array): optimal rejected flow matrix
        actual_node (int): actual node index
        src_node (int): source node index
        dst_node (int): destination node index
        tag (float): tag used to decide the action

    Returns:
        tuple: (action, tag)
    """
    
    src = int(src_node)
    dst = int(dst_node)
    actual = int(actual_node)
    neighbors = list(graph.neighbors(actual))
    
    indices = np.where(np.array(list(graph.edges))[ :,0]==actual)[0]
    prob_to_neighbors = routing_mat[src][dst][indices]
    loss_prob = rejected_mat[src][dst]
    
    #if actual == src: ## see if the packet is rejected
    #    if np.random.rand() <= loss_prob:
    #        return -1, None
        
    if tag:
        if tag in prob_to_neighbors and tag < 1:
            return list(prob_to_neighbors).index(tag), tag
    
    # print(src, dst, actual, neighbors,list(graph.edges), routing_mat.shape, rejected_mat.shape)
    # print(prob_to_neighbors)
    prob_general = list(prob_to_neighbors/sum(prob_to_neighbors))  
    choice = np.random.choice(neighbors, p=prob_general)
    tag = prob_to_neighbors[neighbors.index(choice)]
    return neighbors.index(choice), tag
