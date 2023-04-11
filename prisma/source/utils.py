# -*- coding: utf-8 -*-
from networkx.algorithms.clique import number_of_cliques
import tensorflow as tf
import os, multiprocessing, shutil
import numpy as np
import math

__author__ = "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__license__ = "GPL"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"


def save_model(actor, node_index, path, t, num_episodes, root="saved_models/", snapshot=False):
    """
    Save the DQN model for a node into a folder.

    Parameters
    ----------
    actors : DQN model
        DQN model (for a network node).
    node_index : int
        index of the node.
    path : str
        name of the folder where to store the model.
    t : int
        number of passed train iterations
    num_episodes : int
        number of passed episodes
    root : str, optional
        root folder where to store the model. The default is "saved_models/".
    snapshot : bool, optional
        if True, the model is saved in a folder named "episode_{num_episodes}_step_{t}". The default is False.
    
    Returns
    -------
    None.

    """
    if not(path in os.listdir(root)):
        os.mkdir(root + path)
    path = path.rstrip('/') + '/'
    if snapshot: 
        folder_name = root + path + f"episode_{num_episodes}_step_{t}"
    else:
        folder_name = root + path
    actor.q_network.save(f"{folder_name}/node{node_index}")

def save_all_models(actors, overlay_nodes, path, t, num_episodes, root="saved_models/", snapshot=False):
    """
    Save all DQN models for each node into a folder.

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
    for i in overlay_nodes:
        save_model(actor[i], i, path, t, num_episodes, root, snapshot)
        
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
    from source.models import SplitLayer
    folders = os.listdir(path)
    q_functions = [1]*len(folders)
    for item in os.listdir(path):
        index = int(item.split("_")[-1][4:])
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


def convert_tb_data(root_dir, sort_by=None):
    """Convert local TensorBoard data into Pandas DataFrame.
    
    Function takes the root directory path and recursively parses
    all events data.    
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.
    
    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.
    
    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.
    
    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.
        
    sources : 
        - https://laszukdawid.com/blog/2021/01/26/parsing-tensorboard-data-locally/
        - https://stackoverflow.com/questions/37304461/tensorflow-importing-data-from-a-tensorboard-tfevent-file
    
    """
    import os
    import pandas as pd
    from tensorflow.python.summary.summary_iterator import summary_iterator
    from tensorflow.python.framework import tensor_util

    def convert_tfevent(filepath):
        return pd.DataFrame([
            parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
        ])

    def parse_tfevent(tfevent):
        if "hparams" in tfevent.summary.value[0].tag:
            scalar = 0.0
        else:
            scalar = tensor_util.MakeNdarray(tfevent.summary.value[0].tensor).item()
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=scalar,
        )
    
    columns_order = ['wall_time', 'name', 'step', 'value']
    
    out = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)
        
    return all_df.reset_index(drop=True)