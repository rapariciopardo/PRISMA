#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### imports
import tensorflow as tf
import networkx as nx
import numpy as np
import os
from ns3gym import ns3env
from source.learner import DQN_AGENT
from source.utils import save_model, load_model, LinearSchedule, convert_bps_to_data_rate, optimal_routing_decision
from source.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer, DigitalTwinDB
from source.models import *
import threading
import operator
import copy 
import json
import pandas as pd
import time

__author__ = "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__license__ = "GPL"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"

### define the agent fonction
class Agent():
    """Define the agent action that will be threaded.
    The class will reset the environment, and then retrieve observation to take the proper actions.
    It uses static variables to store global info about the environement
    """
    
    @classmethod
    def init_static_vars(cl, params_dict):
        """Takes the parameters of the simulation from a dict and assign it values to the static vars

        """
        cl.G = params_dict["G"]
        cl.numNodes = params_dict["numNodes"]
        cl.stepTime = params_dict["stepTime"]
        cl.startSim = params_dict["startSim"]
        cl.seed = params_dict["seed"]
        cl.iterationNum = params_dict["iterationNum"]
        cl.prioritizedReplayBuffer=params_dict["prioritizedReplayBuffer"]
        cl.simArgs = {"--simTime": cl.iterationNum,
                    "--testArg": 123}
        cl.debug = 0
        cl.max_out_buffer_size = params_dict["max_out_buffer_size"]
        cl.lr = params_dict["lr"]
        cl.batch_size = params_dict["batch_size"]
        cl.gamma = params_dict["gamma"]
        cl.exploration_initial_eps = params_dict["exploration_initial_eps"]
        cl.exploration_final_eps = params_dict["exploration_final_eps"]
        cl.signaling_type = params_dict["signaling_type"]
        cl.training_step = params_dict["training_step"]
        cl.sync_step = params_dict["sync_step"]
        cl.sync_ratio = params_dict["sync_ratio"]
        cl.replay_buffer_max_size = params_dict["replay_buffer_max_size"]
        cl.traffic_matrix_path = params_dict["traffic_matrix_path"]
        cl.packet_size = params_dict["packet_size"]
        cl.envs = cl.numNodes * [None]
        cl.agents = {i: None for i in range(cl.numNodes)}
        cl.upcoming_events = [[] for n in range(cl.numNodes)]
        cl.throughputs = [pd.DataFrame(columns=("time", "data")) for _ in range(cl.numNodes)]
        if cl.prioritizedReplayBuffer:
            cl.replay_buffer = [PrioritizedReplayBuffer(cl.replay_buffer_max_size, 1, len(list(cl.G.neighbors(n))), n) for n in range(cl.numNodes)]
        else:
            cl.replay_buffer = [ReplayBuffer(cl.replay_buffer_max_size) for n in range(cl.numNodes)]
        ## transition array to be saved for each node
        cl.lock_info_array = [[] for n in range(cl.numNodes)]
        cl.basePort = params_dict["basePort"]
        cl.load_path = params_dict["load_path"]
        cl.logs_folder = params_dict["logs_folder"]
        cl.load_factor = params_dict["load_factor"]
        cl.loss_penalty = params_dict["loss_penalty"]
        cl.link_delay = 0.00#params_dict["link_delay"]
        cl.link_cap = params_dict["link_cap"]
        cl.packet_size = params_dict["packet_size"]
        cl.signalingSim = params_dict["signalingSim"]
        cl.loss_penalty_type = params_dict["loss_penalty_type"]
        cl.nn_max_seg_index = (params_dict["bigSignalingSize"]/cl.packet_size)-1
        cl.total_nb_iterations = 0
        cl.sim_injected_packets=0
        cl.sim_global_injected_packets=0
        cl.sim_dropped_packets = 0
        cl.sim_global_dropped_packets=0
        cl.sim_delivered_packets = 0
        cl.sim_global_delivered_packets = 0
        cl.sim_buffered_packets = 0
        cl.sim_global_buffered_packets = 0
        cl.sim_avg_e2e_delay = 0.0
        cl.sim_sum_e2e_delay = 0.0
        cl.sim_cost = 0.0
        cl.sim_global_avg_e2e_delay = 0.0
        cl.sim_global_sum_e2e_delay = 0.0
        cl.sim_global_cost = 0.0
        cl.sim_bytes_data = 0
        cl.sim_global_bytes_data = 0
        cl.sim_bytes_big_signaling = 0
        cl.sim_bytes_small_signaling = 0
        cl.sim_bytes_overlay_signaling_forward = 0
        cl.sim_bytes_overlay_signaling_back = 0
        cl.total_new_rcv_pkts=0
        cl.total_data_size=0
        cl.total_arrived_pkts=0
        cl.total_e2e_delay=0
        cl.node_lost_pkts=0
        cl.big_signaling_overhead_counter=0
        cl.big_signaling_pkt_counter=0
        cl.small_signaling_overhead_counter=0
        cl.small_signaling_pkt_counter=0
        cl.nb_transitions=0
        cl.curr_time=0
        cl.total_hops=0
        cl.nb_hops=[]
        cl.delays=[]
        cl.delays_ideal=[]
        cl.delays_real=[]
        cl.info_debug=[]
        cl.rewards=[]
        cl.pkt_tracking_dict = {}
        cl.temp_obs = {}
        cl.start_time = time.time()
        cl.nodes_q_network_lock = [threading.Lock() for _ in range(cl.numNodes)]
        cl.nodes_neighbors_copy_lock = [[threading.Lock() for _ in range(cl.numNodes)] for _ in range(cl.numNodes)]
        cl.nodes_target_q_network_lock = [threading.Lock() for _ in range(cl.numNodes)]
        cl.smart_exploration = params_dict["smart_exploration"]
        cl.sessionName=params_dict["session_name"]
        cl.logs_parent_folder = params_dict["logs_parent_folder"]
        cl.total_rewards_with_loss=0
        cl.lambda_train_step = params_dict["lambda_train_step"]
        cl.buffer_soft_limit = params_dict["buffer_soft_limit"]
        cl.lamda_training_start_time = params_dict["lamda_training_start_time"]
        cl.lambda_lr=params_dict["lambda_lr"]
        cl.constrained_loss_database =  [[DigitalTwinDB(Agent.lambda_train_step) for _ in range(len(list(cl.G.neighbors(n))))] for n in range(cl.numNodes)]
        cl.lamda_coefs = [[0 for _ in range(len(list(cl.G.neighbors(n))))] for n in range(cl.numNodes)]
        cl.max_observed_values = [[0 for _ in range(len(list(cl.G.neighbors(n))))] for n in range(cl.numNodes)]
        cl.model_version = params_dict["model_version"]


        cl.sync_counters = [0 for _ in range(cl.numNodes)]
        cl.max_nb_arrived_pkts = params_dict["max_nb_arrived_pkts"]
        if params_dict["agent_type"] == "opt":
            cl.optimal_routing_mat = np.array(json.load(open(params_dict["optimal_soltion_path"]))["routing"])
            cl.optimal_rejected_mat = np.array(json.load(open(params_dict["optimal_soltion_path"]))["rejected_flows"])
            with open("test.txt" , 'wb') as f:
                np.savetxt(f, cl.optimal_rejected_mat, delimiter=' ', newline='\n', header='', footer='', fmt='%1.2f', comments='# ')

    def __init__(self, index, agent_type="dqn", train=True):
        """ Init the agent
        index (int): agent index
        agent_type (str): agent type. Can be : "dqn" for dqn buffer, "dqn_routing" for deep q routing, "sp" for shortest path or "opt" for optimal solution
        train (bool): if true, train the agent. Valid only for agent_type = dqn 
        """
        # check if agent type exists
        if agent_type not in ("dqn_buffer", "dqn_buffer_lite", "dqn_buffer_lighter", "dqn_buffer_lighter_2", "dqn_buffer_lighter_3", "dqn_buffer_ff", "dqn_routing", "dqn_buffer_fp", "dqn_buffer_with_throughputs", "sp", "opt"):
            raise('Unknown agent type, please choose from : ("dqn_buffer", "dqn_buffer_lite", "dqn_buffer_lighter", "dqn_buffer_lighter_2", "dqn_buffer_lighter_3", "dqn_buffer_ff", "dqn_routing", "dqn_buffer_fp", "dqn_buffer_with_throughputs", "sp", "opt")')

        self.agent_type = agent_type
        if "dqn" in agent_type:
            self.train = train
        else:
            self.train = False
        self.index = index

        ### define node neighbors
        self.neighbors = list(Agent.G.neighbors(self.index))
        
        
    
    def _sync_current(self, neighbor_idx, with_temp=False):
        """
        Sync this node neighbor target neural network with the upcoming target nn

        Args:
            neighbor_idx (int): neighbor index for this node
        """
        Agent.agents[self.index].sync_neighbor_target_q_network(neighbor_idx, with_temp=with_temp)


    def run(self):
        pass


