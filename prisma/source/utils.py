# -*- coding: utf-8 -*-
from networkx.algorithms.clique import number_of_cliques
import tensorflow as tf
import os, multiprocessing, shutil
import numpy as np
from source.models import SplitLayer

__author__ = "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__license__ = "GPL"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"


def save_model(actors, path, t, num_episodes, root="saved_models/"):
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
    if(not os.path.exists(root)):
        os.mkdir(root)
    if not(path in os.listdir(root)):
        os.mkdir(root + path)
    path = path.rstrip('/') + '/'
    folder_name = root + path + f"iteration{t}_episode{num_episodes}"
    for i in range(len(actors)):
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