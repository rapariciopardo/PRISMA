#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### imports
from ast import Param
import tensorflow as tf
import networkx as nx
import numpy as np
import os
from ns3gym import ns3env
from source.learner import DQN_AGENT
from source.utils import save_model, load_model, LinearSchedule, convert_bps_to_data_rate, optimal_routing_decision
from source.replay_buffer import ReplayBuffer
from source.models import DQN_buffer_model, DQN_routing_model
import threading
import operator
import copy 
import json
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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
    ## static variables for env description
    envs=[]
    agents = {}
    currIt = 0
    total_new_rcv_pkts=0
    total_arrived_pkts=0
    total_rewards=0
    total_lost_pkts=0
    curr_time=0
    total_hops=0
    total_rewards_with_loss=0
    total_e2e_delay=0
    nb_hops =[]
    delays = []
    delays_ideal =[]
    delays_real = []
    rewards = []
    info_debug = []
    sessionName=None
    # define the replay buffer as a global variable
    replay_buffer = []
    # define the temp observations dict to prepare the (s, a, r, s', flag) for replay buffer
    temp_obs = {}
    # define upcoming events list for each node
    upcoming_events = []
    ## define pkt tracking
    pkt_tracking_dict = {}

    ## general env params
    stepTime=0.1
    startSim=0
    seed=1
    simArgs={}
    debug=0
    iterationNum = 1000
    max_out_buffer_size = 30
    loss_penalty = 0
    max_nb_arrived_pkts = -1
    ## starting port number
    basePort = 0
    ## net topology
    G = None
    numNodes = 0
    ## learning params
    lr = 1e-3
    batch_size = 128
    gamma = 1
    exploration_initial_eps = 0.5
    exploration_final_eps = 0.1
    training_step = 16
    replay_buffer_max_size = 10000
    ## path from where to load the models
    load_path = None
    ## log folder
    logs_folder = "./outputs"
    
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
        cl.replay_buffer = [ReplayBuffer(cl.replay_buffer_max_size) for n in range(cl.numNodes)]
        cl.upcoming_events = [[] for n in range(cl.numNodes)]
        ## transition array to be saved for each node
        cl.lock_info_array = [[] for n in range(cl.numNodes)]
        cl.basePort = params_dict["basePort"]
        cl.load_path = params_dict["load_path"]
        cl.logs_folder = params_dict["logs_folder"]
        cl.loss_penalty = params_dict["loss_penalty"]
        cl.link_delay = 0.00#params_dict["link_delay"]
        cl.link_cap = params_dict["link_cap"]
        cl.packet_size = params_dict["packet_size"]
        cl.currIt = 0
        cl.total_new_rcv_pkts=0
        cl.total_arrived_pkts=0
        cl.total_e2e_delay=0
        cl.total_lost_pkts=0
        cl.curr_time=0
        cl.total_hops=0
        cl.nb_hops=[]
        cl.delays=[]
        cl.delays_ideal=[]
        cl.delays_real=[]
        cl.info_debug=[]
        cl.rewards=[]
        cl.sessionName=params_dict["session_name"]
        cl.total_rewards_with_loss=0
        cl.max_nb_arrived_pkts = params_dict["max_nb_arrived_pkts"]
        if params_dict["agent_type"] == "opt":
            cl.optimal_routing_mat = np.array(json.load(open(params_dict["optimal_soltion_path"]))["routing"])
            cl.optimal_rejected_mat = np.array(json.load(open(params_dict["optimal_soltion_path"]))["rejected_flows"])

    def __init__(self, index, agent_type="dqn", train=True):
        """ Init the agent
        index (int): agent index
        agent_type (str): agent type. Can be : "dqn" for dqn buffer, "dqn_routing" for deep q routing, "sp" for shortest path or "opt" for optimal solution
        train (bool): if true, train the agent. Valid only for agent_type = dqn 
        """
        ### compute the port number
        if agent_type not in ("dqn_buffer", "dqn_routing", "sp", "opt"):
            raise('Unknown agent type, please choose from : ("dqn_buffer", "dqn_routing", "sp", "opt")')

        self.agent_type = agent_type
        if agent_type in ("dqn_buffer", "dqn_routing"):
            self.train = train
        else:
            self.train = False
        self.index = index
        self.port = Agent.basePort + index
        self.stepIdx = 0
        self.sync_step = Agent.sync_step
        ## reset the env
        self._reset()
    
    def _reset(self):
        """ Reset the ns3gym env 
        """

        if Agent.G == None or Agent.numNodes == 0:
            raise("Please make sure you input the topology")

        ### define node neighbors
        self.neighbors = list(Agent.G.neighbors(self.index))


        ### define the ns3 env
        self.env = ns3env.Ns3Env(port=int(self.port), stepTime=Agent.stepTime, startSim=Agent.startSim, simSeed=Agent.seed, simArgs=Agent.simArgs, debug=Agent.debug)
        self.obs = self.env.reset()
        Agent.envs[self.index] = self.env

        ## define the agent
        if self.agent_type == "dqn_buffer":
            ## declare the DQN buffer model
            Agent.agents[self.index] = DQN_AGENT(
                q_func=DQN_buffer_model,
                # observation_shape=self.env.observation_space.shape,
                observation_shape=self.env.observation_space.shape,
                num_actions=self.env.action_space.n,
                num_nodes=Agent.numNodes,
                input_size_splits = [1,
                                    self.env.action_space.n,
                                    ],
                lr=Agent.lr,
                gamma=Agent.gamma,
                neighbors_degrees=[len(list(Agent.G.neighbors(x))) for x in self.neighbors]
            )
        elif self.agent_type == "dqn_routing":
            ## declare the DQN buffer model
            Agent.agents[self.index] = DQN_AGENT(
                q_func=DQN_routing_model,
                # observation_shape=self.env.observation_space.shape,
                observation_shape=self.env.observation_space.shape,
                num_actions=self.env.action_space.n,
                num_nodes=Agent.numNodes,
                input_size_splits = [1,
                                    self.env.action_space.n,
                                    ],
                lr=Agent.lr,
                gamma=Agent.gamma,
                neighbors_degrees=[len(list(Agent.G.neighbors(x))) for x in self.neighbors]
            )
        elif self.agent_type == "opt":
            Agent.agents[self.index] = optimal_routing_decision
        else:
            Agent.agents[self.index] = nx.shortest_path

        ## compute big signaling delay
        if self.agent_type in ("dqn_buffer", "dqn_routing"):
            self.nn_size = np.sum([np.prod(x.shape) for x in Agent.agents[self.index].q_network.trainable_weights])*32
            self.big_signaling_delay = (self.nn_size/ Agent.link_cap) + Agent.link_delay
        
        ### compute small signaling delay
        if Agent.signaling_type == "NN":
            self.small_signaling_pkt_size = 64 + 8 + (8 * (len(self.neighbors)+1)) # header + reward (float) + s' (double)
            self.small_signaling_delay = (self.small_signaling_pkt_size / Agent.link_cap) + Agent.link_delay
            if self.sync_step < 0:
                self.sync_step = self._compute_sync_step(ratio=Agent.sync_ratio)
        elif Agent.signaling_type == "target":
            self.small_signaling_pkt_size = 64 + 8  # header + target (float)
            self.small_signaling_delay = (self.small_signaling_pkt_size / Agent.link_cap) + Agent.link_delay
            self._sync_all() # intialize target networks

        ## load the models
        if Agent.load_path is not None and self.agent_type in ("dqn_buffer", "dqn_routing"):
            loaded_models = load_model(Agent.load_path, self.index)
            if loaded_models is not None:
                Agent.agents[self.index].q_network.set_weights(loaded_models[self.index].get_weights())
                print("Restoring from {}".format(Agent.load_path))

        # Create the schedule for exploration.
        self.exploration = LinearSchedule(schedule_timesteps=int(Agent.iterationNum),
                                    initial_p=Agent.exploration_initial_eps,
                                    final_p=Agent.exploration_final_eps)

        ## env trackers definition
        self.episode_mean_td_error = []
        self.count_arrived_packets = 0
        self.count_new_pkts = 0
        self.last_training_time = 0
        self.last_sync_time = 0
        self.last_training_step = 0
        self.gradient_step_idx = 0

        ## define the log file for td error and exploration value
        self.summary_writer_td_error = tf.summary.create_file_writer(logdir=f'{Agent.logs_folder}/td_error/node_{self.index}')
        self.summary_writer_exploration = tf.summary.create_file_writer(logdir=f'{Agent.logs_folder}/exploration/node_{self.index}')
        self.summary_writer_replay_buffer_length = tf.summary.create_file_writer(logdir=f'{Agent.logs_folder}/replay_buffer_length/node_{self.index}')

    def _get_reward(self):
        """ Compute the reward when the packet is lost
        """
        return Agent.loss_penalty
    
    def _take_action(self, obs):
        """ Take an action given self.obs

        Args :
            obs (list): observation list
        """
        if self.agent_type in("dqn_buffer", "dqn_routing"):
            ### Take action using the NN
            action = Agent.agents[self.index].step(np.array([obs]), self.train, self.update_eps).numpy().item()
        elif self.agent_type == "sp":
            action = self.neighbors.index(Agent.agents[self.index](Agent.G, self.index, obs[0])[1])
        elif self.agent_type == "opt":
            track =  Agent.pkt_tracking_dict[int(self.pkt_id)]
            action, track["tag"] = optimal_routing_decision(Agent.G, Agent.optimal_routing_mat, Agent.optimal_rejected_mat, self.index, track["src"], track["dst"], track["tag"])
        return action

    def _forward(self):
        """
        Do an env step
        """
        ## schedule the exploration
        self.update_eps = tf.constant(self.exploration.value(self.stepIdx))
        
        ## take the action
        if self.obs[0] == self.index or self.stepIdx < 1: # pkt arrived to dst or it is a train step, ignore the action
            self.action = 0
        else:
            self.action = self._take_action(self.obs)
            ## check if the pkt is lost
            if self.obs[self.action + 1] >= Agent.max_out_buffer_size or self.action == -1:
                Agent.total_lost_pkts += 1
                rew = self._get_reward()
                Agent.total_rewards_with_loss += rew
                Agent.rewards.append(rew)
                if self.train:
                    next_hop_degree = list(Agent.G.neighbors(self.neighbors[self.action]))
                    Agent.replay_buffer[self.index].add(np.array(self.obs, dtype=float).squeeze(),
                                        self.action, 
                                        rew,
                                        np.array([self.obs[0]] + [0]*len(next_hop_degree), dtype=float).squeeze(), 
                                        True)
                if  self.action == -1:
                    self.action =Agent.numNodes
                
            ## Add to the temp obs
            else:
                Agent.temp_obs[int(self.pkt_id)]= {"node": self.index,
                                                   "obs": self.obs,
                                                   "action": self.action,
                                                   "time": Agent.curr_time, 
                                                   "src" :Agent.pkt_tracking_dict[int(self.pkt_id)]["src"],
                                                   "dst" :Agent.pkt_tracking_dict[int(self.pkt_id)]["dst"],
                                                   }

        ### Apply the action
        return self.env.step(self.action)

    def _sync(self, neighbor_num, neighbor_idx):
        """
        Sync this node neighbor target neural network

        Args:
            neighbor_num (int): neighbor number
            neighbor_idx (int): neighbor index for this node
        """
        #print("aqui2")
        #print(Agent.agents[neighbor_num])
        Agent.agents[self.index].sync_neighbor_target_q_network(Agent.agents[neighbor_num], neighbor_idx)

    def _sync_all(self):
        """
        Sync this node all neighbors neural networks.
        If signaling type is target, sync the node target NN.
        """
        if self.signaling_type == "target":
            Agent.agents[self.index].update_target()
        else:
            #print("aqui2")
            for indx, neighbor in enumerate(self.neighbors): 
                self._sync(neighbor, indx)

    def _check_sync(self):
        """
        Check the time to sync the NN depending on the signaling mode
        """
        ### Sync target NN
        if Agent.signaling_type in ("ideal", "target"):
            if Agent.curr_time > (self.last_sync_time + self.sync_step):
                self._sync_all()
                self.last_sync_time = Agent.curr_time
                
        elif Agent.signaling_type == "NN":
            #print("aqui")
            if Agent.curr_time > (self.last_sync_time + self.big_signaling_delay + self.sync_step):
                self._sync_all()
                self.last_sync_time = Agent.curr_time

    def _compute_sync_step(self, ratio=0.1):
        """
        Compute sync step to have control over data of ratio.
        """
        ## load traffic matrix and convert it to bps
        traff_mat = np.loadtxt(Agent.traffic_matrix_path, dtype=object)
        traff_mat = np.vectorize(convert_bps_to_data_rate)(traff_mat)
        
        
        ## data load per second
        data_load_per_s = np.sum(traff_mat)

        ## number of pkts per second
        nb_pkts_per_s = data_load_per_s/ (Agent.packet_size*8)

        ## small signaling load per s
        control_load_per_s = (nb_pkts_per_s * self.small_signaling_pkt_size)

        ## compute sync step
        sync_step = (self.nn_size) /((data_load_per_s * ratio)- control_load_per_s)
        print(f"Sync step computed automatically : {sync_step} seconds")
        return sync_step

    def _get_upcoming_events(self):
        """
        Go through the upcoming event list and check whether of signaling pkt arrived, then add it to replay buffer
        """
        if Agent.signaling_type == "NN":
            for element in Agent.upcoming_events[self.index]:
                if element["time"] > Agent.curr_time:
                    break
                Agent.replay_buffer[self.index].add(element["obs"],
                                                element["action"], 
                                                element["reward"],
                                                element["new_obs"], 
                                                element["flag"])
        elif Agent.signaling_type == "target":
            for element in Agent.upcoming_events[self.index]:
                if element["time"] > Agent.curr_time:
                    break
                Agent.replay_buffer[self.index].add(element["obs"],
                                                element["action"], 
                                                element["target"],
                                                element["new_obs"], 
                                                element["flag"])

    def _train(self):
        """
        Do a training step
        """
        self.last_training_time = Agent.curr_time
        self.last_training_step = self.stepIdx
        # if self.gradient_step_idx ==0 and self.index == 8:
        #     indices = np.where(np.stack(np.array(Agent.replay_buffer[self.index]._storage)[:,0], axis=0)[:, 0] == 4.)[0]
        #     np.savetxt(f"savings/replay_buffer{self.index}.txt", np.array(Agent.replay_buffer[self.index]._storage, dtype=object)[indices], "%s")
        # print("train...", index)
        ## sample from the replay buffer
        obses_t, actions_t, rewards_t, next_obses_t, dones_t = Agent.replay_buffer[self.index].sample(Agent.batch_size)
        weights, _ = np.ones(Agent.batch_size, dtype=np.float32), None

        if Agent.signaling_type == "target":
            targets_t = tf.constant(rewards_t, dtype=float)
            obses_t = tf.constant(obses_t)
            actions_t = tf.constant(actions_t)
        else:
            ### Construct the target values
            targets_t = []
            action_indices_all = []
            for indx, neighbor in enumerate(self.neighbors):
                filtered_indices = np.where(np.array(list(Agent.G.neighbors(neighbor)))!=self.index)[0] # filter the net interface from where the pkt comes
                action_indices = np.where(actions_t == indx)[0]
                action_indices_all.append(action_indices)
                if len(action_indices):
                    targets_t.append(Agent.agents[self.index].get_neighbor_target_value(indx, rewards_t[action_indices], tf.constant(
                        np.array(np.vstack(next_obses_t[action_indices]), dtype=float)), dones_t[action_indices], filtered_indices))
            action_indices_all = np.concatenate(action_indices_all)

            ### prepare tf variables
            obses_t = tf.constant(obses_t[action_indices_all,])
            actions_t = tf.constant(actions_t[action_indices_all], shape=(Agent.batch_size))
            targets_t = tf.constant(tf.concat(targets_t, axis=0), shape=(Agent.batch_size))
        
        weights = tf.constant(weights)

        ### Make a gradient step
        td_errors = Agent.agents[self.index].train(obses_t, actions_t, targets_t, weights)    
        self.episode_mean_td_error.append(np.mean(td_errors))
        # print(self.index, Agent.curr_time, self.gradient_step_idx, np.mean(td_errors))
        with self.summary_writer_td_error.as_default():
            tf.summary.scalar('MSE_loss_over_steps', np.mean(td_errors**2), step=self.gradient_step_idx)
            tf.summary.scalar('MSE_loss_over_time', np.mean(td_errors**2), step=int(Agent.curr_time*1e6))
        with self.summary_writer_exploration.as_default():
            tf.summary.scalar('exploaration_value_over_steps', self.update_eps, step=self.gradient_step_idx)
            tf.summary.scalar('exploaration_value_over_time', self.update_eps, step=int(Agent.curr_time*1e6))
        with self.summary_writer_replay_buffer_length.as_default():
            tf.summary.scalar('replay_buffer_length_over_steps', len(Agent.replay_buffer[self.index]), step=self.gradient_step_idx)
            tf.summary.scalar('replay_buffer_length_over_time', len(Agent.replay_buffer[self.index]), step=int(Agent.curr_time*1e6))
        self.gradient_step_idx += 1

    def run_forwarder(self):
        """ Run an episode simulation
        """
        try:
            while True:
                self.obs = self.env.reset()
                self.stepIdx = 0
                while True:
                    if(not self.env.connected):
                        break
                    self.obs, _, self.done, self.info = self._forward()
                    
                    ## check if episode is done
                    if self.done and self.obs[0] == -1:
                        break
                    
                    ## Increment the simulation and episode counters
                    self.stepIdx += 1
                    Agent.currIt += 1

                    ## info treatments
                    tokens = self.info.split(",")
                    delay_time = float(tokens[0].split('=')[-1])
                    Agent.curr_time = float(tokens[4].split('=')[-1])
                    self.pkt_id = float(tokens[5].split('=')[-1])

                    if self.pkt_id not in Agent.pkt_tracking_dict.keys(): ## check if the packet is a new arrival
                        self.count_new_pkts += 1
                        Agent.total_new_rcv_pkts += 1
                        ## add to tracked pkts
                        Agent.pkt_tracking_dict[int(self.pkt_id)]= {"src": self.index,
                                                                    "node": self.index,
                                                                    "dst": int(self.obs[0]),
                                                                    "hops": [self.index],
                                                                    "delays_ideal": [],
                                                                    "delays_real": [],
                                                                    "start_time": Agent.curr_time,
                                                                    "tag": None}
                    else: ## if the packet is not new in the network
                        states_info = Agent.temp_obs.pop(self.pkt_id)
                        hop_time_real =  Agent.curr_time - states_info["time"]
                        hop_time_ideal = ((states_info["obs"][states_info["action"] + 1] +1 ) * Agent.packet_size * 8 / Agent.link_cap) + Agent.link_delay
                        Agent.total_rewards_with_loss += hop_time_real
                        ## add to tracked pkts
                        Agent.pkt_tracking_dict[int(self.pkt_id)]["hops"].append(self.index)                        
                        Agent.pkt_tracking_dict[int(self.pkt_id)]["node"] = self.index
                        
                        ## saving state info
                        states_info["hop_time_real"] = hop_time_real
                        states_info["hop_time_ideal"] = hop_time_ideal
                        Agent.rewards.append(hop_time_ideal)
                        Agent.pkt_tracking_dict[int(self.pkt_id)]["delays_ideal"].append(hop_time_ideal)
                        Agent.pkt_tracking_dict[int(self.pkt_id)]["delays_real"].append(hop_time_real)                  
      
                        curr_node_to_save= int(states_info["node"])
                        Agent.lock_info_array[curr_node_to_save].append([int(states_info["src"]),
                                                                   int(states_info["dst"]),
                                                                   curr_node_to_save,
                                                                   list(Agent.G.neighbors(curr_node_to_save))[states_info["action"]],
                                                                   hop_time_ideal,
                                                                   hop_time_real,
                                                                   states_info["obs"],
                                                                   states_info["action"]
                                                                        ])
                        
                        
                        if Agent.signaling_type == "ideal":
                            Agent.replay_buffer[int(states_info["node"])].add(np.array(states_info["obs"], dtype=float).squeeze(),
                                                                        states_info["action"], 
                                                                        hop_time_ideal,
                                                                        np.array(self.obs, dtype=float).squeeze(), 
                                                                        self.done)
                        elif Agent.signaling_type == "NN":
                            self.upcoming_events[int(states_info["node"])].append({ "time": Agent.curr_time + self.small_signaling_delay,
                                                                                    "obs" : np.array(states_info["obs"], dtype=float).squeeze(),
                                                                                    "action": states_info["action"], 
                                                                                    "reward": hop_time_ideal,
                                                                                    "new_obs": np.array(self.obs, dtype=float).squeeze(), 
                                                                                    "flag": self.done
                                                                                    })
                            self.upcoming_events[int(states_info["node"])].sort(key=operator.itemgetter('time'))

                        elif Agent.signaling_type == "target":
                            ## compute the target value
                            filtered_index = np.where(np.array(list(Agent.G.neighbors(self.index)))!=int(states_info["node"]))[0] # filter the net interface from where the pkt comes 
                            target = Agent.agents[self.index].get_target_value(np.array([hop_time_ideal]), np.array([self.obs]), np.array([self.done]), filtered_index)
                            # target = hop_time_ideal + Agent.gamma * (1- int(self.done)) * tf.reduce_min(Agent.agents[self.index].q_network(np.array([self.obs], dtype=float)), 1)
                            self.upcoming_events[int(states_info["node"])].append({ "time": Agent.curr_time + self.small_signaling_delay,
                                                        "obs" : np.array(states_info["obs"], dtype=float).squeeze(),
                                                        "action": states_info["action"], 
                                                        "target": target.numpy().item(),
                                                        "new_obs": np.array(self.obs, dtype=float).squeeze(), 
                                                        "flag": self.done
                                                        })
                            self.upcoming_events[int(states_info["node"])].sort(key=operator.itemgetter("time"))
                        
                        if self.done: ## if the packet arrived to destination  
                            self.count_arrived_packets += 1
                            Agent.total_arrived_pkts += 1
                            # Agent.total_e2e_delay += delay_time
                            hops =  len(Agent.pkt_tracking_dict[int(self.pkt_id)]["hops"]) - 1
                            # Agent.total_e2e_delay += Agent.curr_time - Agent.pkt_tracking_dict[int(self.pkt_id)]["start_time"]
                            Agent.total_hops += hops
                            Agent.total_e2e_delay += delay_time
                            
                            Agent.delays_ideal.append(sum(Agent.pkt_tracking_dict[int(self.pkt_id)]["delays_ideal"]))
                            Agent.delays_real.append(sum(Agent.pkt_tracking_dict[int(self.pkt_id)]["delays_real"]))
                            
                            Agent.delays.append(delay_time)
                            Agent.info_debug.append([Agent.pkt_tracking_dict[int(self.pkt_id)]["src"], Agent.pkt_tracking_dict[int(self.pkt_id)]["dst"], Agent.pkt_tracking_dict[int(self.pkt_id)]["hops"], len(Agent.pkt_tracking_dict[int(self.pkt_id)]["hops"])-1, delay_time])
                            Agent.total_hops += hops
                            Agent.nb_hops.append(hops)
                            if(len(Agent.nb_hops)>50):
                                Agent.nb_hops = Agent.nb_hops[-50:]
                            if(len(Agent.delays)>50):
                                Agent.delays = Agent.delays[-50:]
                            Agent.pkt_tracking_dict.pop(int(self.pkt_id))
                            if Agent.max_nb_arrived_pkts > 0 and Agent.max_nb_arrived_pkts <= Agent.total_arrived_pkts:
                                print("Done by max number of arrived pkts")
                                break
                break

        except KeyboardInterrupt:
            print("index :", self.index, "Ctrl-C -> Exit")
            self.env.close()
        if(not os.path.exists("logs/")):
            os.mkdir("logs")
        np.savetxt("logs/log_dict_"+Agent.sessionName+".txt", np.asarray(Agent.info_debug, dtype='object'), fmt='%s')
        self.env.ns3ZmqBridge.send_close_command()
        # print("***index :", self.index, "Done", "stepIdx =", self.stepIdx, "arrived pkts =", self.count_arrived_packets,  "new received pkts", self.count_new_pkts, "gradient steps", self.gradient_step_idx)
        return True

    def run_trainer(self, train_type="event"):
        """
            train the agent at training_step depending on the train type.
            Args :
                train_type (str) : can be "event" for event based triggering or "time" for time based triggering
        """
        import time
        if train_type == "event":
            while True :
                time.sleep(np.random.uniform(0.1, 0.7))
                ## check if there are signaling pkts arrived
                if Agent.signaling_type in ("NN", "target"):
                    self._get_upcoming_events()
                if self.stepIdx > (self.last_training_step + Agent.training_step) and len(Agent.replay_buffer[self.index])> Agent.batch_size:
                    self._check_sync()
                    self._train()
        elif train_type == "time":
            while True :
                time.sleep(np.random.uniform(0.1, 0.7))
                ## check if there are signaling pkts arrived if signaling type NN
                if Agent.signaling_type in ("NN", "target"):
                    self._get_upcoming_events()
                ## check if it is time to train
                if Agent.curr_time > (self.last_training_time + Agent.training_step) and len(Agent.replay_buffer[self.index])> Agent.batch_size:
                    #print(len(Agent.replay_buffer[self.index]))
                    self._check_sync()
                    self._train()
