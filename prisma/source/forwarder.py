### imports
import tensorflow as tf
import networkx as nx
import numpy as np
from ns3gym import ns3env
from source.learner import DQN_AGENT
from source.utils import load_model, LinearSchedule, optimal_routing_decision
from source.models import *
import operator
import pandas as pd
import time
from source.agent import Agent

__author__ = "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__license__ = "GPL"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"

class Forwarder(Agent):
    """Forwarder agent class.
    """
    def __init__(self, index, agent_type="dqn", train=True):
        """Initialize the forwarder agent.
            Args:
                index (int): agent index.
                agent_type (str): agent type.
                train (bool): train or test mode.
        """
        ## initialize the parent class
        Agent.__init__(self, index, agent_type, train)
        
        ## define the communication port
        self.port = Agent.basePort + index
        print("Index: ", self.index, "Port: ", self.port)
        self.transition_number = 0
        
        ## reset the env
        self.reset()
        
    def reset(self, init=True):
        """ Reset the ns3gym env 
        """
        if Agent.G == None or Agent.numNodes == 0:
            raise("Please make sure you input the topology")

        ### define the ns3 env
        self.env = ns3env.Ns3Env(port=int(self.port), stepTime=Agent.stepTime, startSim=Agent.startSim, simSeed=Agent.seed, simArgs=Agent.simArgs, debug=Agent.debug)
        obs = self.env.reset()
        Agent.envs[self.index] = self.env
        if init:
            ## define the agent
            if self.agent_type == "dqn_buffer":
                ## declare the DQN buffer model
                model = DQN_buffer_model
            elif self.agent_type == "dqn_buffer_lite":
                ## declare the DQN buffer lite model
                model = DQN_buffer_lite_model
            elif self.agent_type == "dqn_buffer_lighter":
                ## declare the DQN buffer lighter model
                model = DQN_buffer_lighter_model    
            elif self.agent_type == "dqn_buffer_lighter_2":
                ## declare the DQN buffer lighter_2 model
                model = DQN_buffer_lighter_2_model
            elif self.agent_type == "dqn_buffer_lighter_3":
                ## declare the DQN buffer lighter_3 model
                model = DQN_buffer_lighter_3_model
            elif self.agent_type == "dqn_buffer_ff":
                ## declare the DQN buffer ff model
                model = DQN_buffer_ff_model
            elif self.agent_type == "dqn_routing":
                ## declare the DQN buffer model
                model = DQN_routing_model
                
            if "dqn" in self.agent_type:
                Agent.agents[self.index] = DQN_AGENT(
                    q_func=model,
                    observation_shape=self.env.observation_space.shape,
                    num_actions=self.env.action_space.n,
                    num_nodes=Agent.numNodes,
                    input_size_splits = [1,
                                        self.env.action_space.n,
                                        ],
                    lr=Agent.lr,
                    gamma=Agent.gamma,
                    neighbors_degrees=[len(list(Agent.G.neighbors(x))) for x in self.neighbors],
                    d_t_max_time=Agent.d_t_max_time,
                    d_q_func=model,


                )
            elif self.agent_type == "opt":
                Agent.agents[self.index] = optimal_routing_decision
            elif self.agent_type == "sp":
                Agent.agents[self.index] = nx.shortest_path
            else:
                raise ValueError("Unknown agent type")

            ## compute big signaling delay
            if "dqn" in self.agent_type:
                self.nn_size = np.sum([np.prod(x.shape) for x in Agent.agents[self.index].q_network.trainable_weights])*32
                self.big_signaling_delay = (self.nn_size/ Agent.link_cap) + Agent.link_delay
                print("node:", self.index, "big signaling delay: ", self.big_signaling_delay, self.nn_size)
        
            ### compute small signaling delay
            if Agent.signaling_type == "NN":
                self.small_signaling_pkt_size = 64 + 8 + (8 * (len(self.neighbors)+1)) # header + reward (float) + s' (double)
                self.small_signaling_delay = (self.small_signaling_pkt_size / Agent.link_cap) + Agent.link_delay
                if self.sync_step < 0:
                    self.sync_step = self._compute_sync_step(ratio=Agent.sync_ratio)
            elif Agent.signaling_type == "target":
                self.small_signaling_pkt_size = 64 + 8  +8 # header + target (float)
                self.small_signaling_delay = (self.small_signaling_pkt_size / Agent.link_cap) + Agent.link_delay
            elif Agent.signaling_type == "digital_twin":
                self.small_signaling_pkt_size = 64 + (8 * len(self.neighbors)) + (8 * (len(self.neighbors)+1)) + 8 # header + targets vector (float) +  s' vector (float) + current time (double)
                self.small_signaling_delay = (self.small_signaling_pkt_size / Agent.link_cap) + Agent.link_delay
                # self._sync_all() # intialize target networks

            ## load the dt models
            if Agent.load_path is not None and "dqn" in self.agent_type:
                if Agent.signaling_type == "digital_twin" and self.train: # load digital twin and the model
                    d_t_loaded_models = load_model(Agent.load_path, -1)
                    for neighbor_idx, neighbor in enumerate(self.neighbors):
                        Agent.agents[self.index].neighbors_d_t_network[neighbor_idx].set_weights(d_t_loaded_models[neighbor].get_weights())
                    print("Restoring Digital Twin from {} for node {}".format(Agent.load_path, self.index))                    

            ## load the models
            if Agent.load_path is not None and "dqn" in self.agent_type:
                # load the model
                loaded_models = load_model(Agent.load_path, self.index)
                if loaded_models is not None:
                    print("Restoring from {} for node {}".format(Agent.load_path, self.index))
                    Agent.agents[self.index].q_network.set_weights(loaded_models[self.index].get_weights())
        
            ## define the log file for exploration value
            self.tb_writer_dict = {"exploration":  tf.summary.create_file_writer(logdir=f'{Agent.logs_folder}/exploration/node_{self.index}')}
            
        ## env trackers definition
        self.count_arrived_packets = 0
        self.count_new_pkts = 0
        self.update_eps = 0
        Agent.sync_counters[self.index] = -1
        
        ## define action history and nb_actions
        self.action_history = np.ones(self.env.action_space.n)
        self.nb_actions = np.sum(self.action_history)
            

        # Create the schedule for exploration.
        self.exploration = LinearSchedule(schedule_timesteps=int(Agent.iterationNum),
                                    initial_p=Agent.exploration_initial_eps,
                                    final_p=Agent.exploration_final_eps)



    def step(self, obs):
        """
        Do an env step
        
        """
        ## schedule the exploration
        if self.train:
            self.update_eps = tf.constant(self.exploration.value(self.transition_number))
            ## log the exploration value
            with self.tb_writer_dict["exploration"].as_default():
                tf.summary.scalar('exploaration_value_over_steps', self.update_eps, step=self.transition_number)
                tf.summary.scalar('exploaration_value_over_time', self.update_eps, step=int((Agent.base_curr_time + Agent.curr_time)*1e6))

        ## take the action
        if obs[0] == self.index or self.transition_number < 1 or self.signaling == True or obs[0] in(-1, 1000): # pkt arrived to dst or it is a train step, ignore the action
            self.action = 0
        else:
            self.action = self.take_action(obs)
            Agent.temp_obs[int(self.pkt_id)]= {"node": self.index,
                                               "obs": obs,
                                               "action": self.action,
                                               "time": Agent.curr_time,
                                               "src" :Agent.pkt_tracking_dict[int(self.pkt_id)]["src"],
                                               "dst" :Agent.pkt_tracking_dict[int(self.pkt_id)]["dst"],
                                               }
            # Agent.max_observed_values[self.index] = list(np.max([Agent.max_observed_values[self.index], obs[1:]], axis=0))
            if Agent.loss_penalty_type == "constrained":
                Agent.constrained_loss_database[self.index][self.action].add(obs[self.action],
                                                                    Agent.lamda_coefs[self.index][self.action],
                                                                    Agent.curr_time)
        ### Apply the action
        return self.env.step(self.action)
    
    def _get_reward_lost_pkt(self):
        """ Compute the reward when the packet is lost
        """
        if Agent.loss_penalty_type == "fixed":
            return Agent.loss_penalty


    def take_action(self, obs):
        """ Take an action given obs

        Args :
            obs (list): observation list
        """
        if "dqn" in self.agent_type:
            actions_probs = None
            if Agent.smart_exploration:
                actions_probs = 1-(self.action_history/self.nb_actions)
                actions_probs /=sum(actions_probs)
            ### Take action using the NN
            action = Agent.agents[self.index].step(np.array([obs]),
                                                   self.train,
                                                   self.update_eps,
                                                   actions_probs=actions_probs).numpy().item()
            if Agent.smart_exploration:
                self.action_history[action] += 1
                self.nb_actions += 1
        elif self.agent_type == "sp":
            action = self.neighbors.index(Agent.agents[self.index](Agent.G, self.index, obs[0])[1])
        elif self.agent_type == "opt":
            track =  Agent.pkt_tracking_dict[int(self.pkt_id)]
            action, track["tag"] = optimal_routing_decision(Agent.G, Agent.optimal_routing_mat, Agent.optimal_rejected_mat, self.index, track["src"], track["dst"], track["tag"])
        return action
    
    def treat_info(self, info):
        """ Treat the info received from the ns3 simulator
        Args:
            info (dict): info received from the ns3 simulator
        Returns:
            bool: True if it is a control packet, False otherwise
        """
        tokens = info.split(",")
        self.delay_time = float(tokens[0].split('=')[-1])
        ## retrieve packet info
        self.pkt_size = float(tokens[1].split('=')[-1])
        Agent.curr_time = float(tokens[2].split('=')[-1])
        self.pkt_id = int(tokens[3].split('=')[-1])
        pkt_type = int(tokens[4].split('=')[-1]) 
        self.signaling = pkt_type != 0 
        if(pkt_type==0): # data packet
            # treat lost packets
            lost_packets_id = tokens[18].split('=')[-1].split(';')[:-1] 
            for lost_packet_id in lost_packets_id: 
                lost_packet_info = Agent.temp_obs.get(int(lost_packet_id)) 
                if(lost_packet_info==None): 
                    print("error") 
                    continue 
                #if(int(lost_packet_time)!= int(lost_packet_info["time"]*1000)): 
                #    continue 
                next_hop_degree = len(list(Agent.G.neighbors(self.neighbors[lost_packet_info["action"]])))
                rew = self._get_reward_lost_pkt()
                obs_shape = next_hop_degree
                ## Add the lost packet to the replay buffer
                if Agent.loss_penalty_type == "fixed":
                    if(Agent.prioritizedReplayBuffer):
                        Agent.replay_buffer[self.index].add(np.array(lost_packet_info["obs"], dtype=float).squeeze(),
                                    lost_packet_info["action"], 
                                    rew,
                                    np.array([lost_packet_info["obs"][0]] + [0]*(obs_shape), dtype=float).squeeze(), 
                                    True,
                                    Agent.replay_buffer[self.index].latest_gradient_step[lost_packet_info["action"]])
                    else:
                        if(self.train):
                            Agent.replay_buffer[self.index].add(np.array(lost_packet_info["obs"], dtype=float).squeeze(),
                                        lost_packet_info["action"], 
                                        rew,
                                        np.array([lost_packet_info["obs"][0]] + [0]*(obs_shape), dtype=float).squeeze(), 
                                        True)
                ## Increment the loss counter
                Agent.node_lost_pkts += 1
                ## Remove the lost packet from the pkt tracking dict
                Agent.pkt_tracking_dict.pop(int(lost_packet_id))
        else: 
            if(pkt_type==2): # small signaling packet
                id_signaled = int(tokens[18].split('=')[-1]) 
                self._get_upcoming_events_real(id_signaled) 
                Agent.small_signaling_overhead_counter += self.pkt_size 
                Agent.small_signaling_pkt_counter += 1 
            if(pkt_type==1): # big signaling packet
                NNIndex = int(tokens[18].split('=')[-1]) 
                segIndex= int(tokens[19].split('=')[-1]) 
                NodeIdSignaled = int(tokens[20].split('=')[-1]) 
                if segIndex > Agent.nn_max_seg_index:
                    raise("segIndex > {}".format(Agent.nn_max_seg_index))
                if segIndex == Agent.nn_max_seg_index: ## NN signaling complete
                    # print(f"sync {self.index} with neighbor {self.neighbors.index(NodeIdSignaled)} at time {Agent.curr_time} {self.sync_counter} {NNIndex}")
                    if NNIndex ==Agent.sync_counters[self.index] - 1:
                        self._sync_current(self.neighbors.index(NodeIdSignaled), with_temp=True)
                    else:
                        #print(self.index, NodeIdSignaled)
                        self._sync_current(self.neighbors.index(NodeIdSignaled))
                #print("here") 
                Agent.big_signaling_overhead_counter += self.pkt_size 
                Agent.big_signaling_pkt_counter += 1

            return True
            #continue             
        
        ## update stats in static variables
        Agent.sim_avg_e2e_delay =  float(tokens[5].split('=')[-1])  
        Agent.sim_cost = float(tokens[6].split('=')[-1]) 
        Agent.sim_global_avg_e2e_delay = float(tokens[7].split('=')[-1])  
        Agent.sim_global_cost = float(tokens[8].split('=')[-1])  
        Agent.sim_dropped_packets = float(tokens[9].split('=')[-1]) 
        Agent.sim_delivered_packets = float(tokens[10].split('=')[-1]) 
        Agent.sim_injected_packets = float(tokens[11].split('=')[-1]) 
        Agent.sim_buffered_packets = float(tokens[12].split('=')[-1]) 
        Agent.sim_global_dropped_packets = float(tokens[13].split("=")[-1]) + Agent.sim_dropped_packets 
        Agent.sim_global_delivered_packets = float(tokens[14].split("=")[-1]) + Agent.sim_delivered_packets 
        Agent.sim_global_injected_packets = float(tokens[15].split("=")[-1]) + Agent.sim_injected_packets 
        Agent.sim_global_buffered_packets = float(tokens[16].split('=')[-1]) + Agent.sim_buffered_packets 
        Agent.sim_signaling_overhead = float(tokens[17].split('=')[-1])
        if Agent.sim_global_delivered_packets > 0:
            Agent.sim_global_avg_e2e_delay = ((Agent.sim_global_avg_e2e_delay * float(tokens[13].split("=")[-1])) + (Agent.sim_avg_e2e_delay * Agent.sim_delivered_packets))/(Agent.sim_global_delivered_packets)
        if Agent.sim_global_delivered_packets + Agent.sim_global_dropped_packets > 0:
            Agent.sim_global_cost = ((Agent.sim_global_cost * (float(tokens[13].split("=")[-1]) + float(tokens[13].split("=")[-1]))) + (Agent.sim_cost * (Agent.sim_dropped_packets + Agent.sim_delivered_packets)))/(Agent.sim_global_dropped_packets + Agent.sim_global_delivered_packets)
        return False

    def run(self):
        """ 
        Run an episode simulation
        """
        while True:
            obs = self.env.reset()
            self.transition_number = 0
                
            while True:
                if(not self.env.connected):
                    break
                obs, _, done_flag, info = self.step(obs)
                ## check if episode is done_flag
                if done_flag and obs[0] == -1:
                    break

                ## Treat the info from the env
                if self.treat_info(info):
                    continue # if it is a control packet, continue

                ## Increment the simulation and episode counters
                self.transition_number += 1
                Agent.total_nb_iterations += 1
                
                Agent.nb_transitions += 1
                if self.pkt_id not in Agent.pkt_tracking_dict.keys(): ## check if the packet is a new arrival
                    self.handle_new_packet(obs)
                    
                else: ## if the packet is not new in the network
                    self.handle_transit_packet(obs, done_flag)

                    if done_flag: ## if the packet arrived to destination
                        self.handle_done()
                        
                        ## check if the episode is done by max number of arrived pkts
                        if Agent.max_nb_arrived_pkts > 0 and Agent.max_nb_arrived_pkts <= Agent.total_arrived_pkts:
                            print("Done by max number of arrived pkts")
                            break
            break
        ## close the zmq bridge
        self.env.ns3ZmqBridge.send_close_command()
        return True
    
    def handle_new_packet(self, obs):
        """ Handle a new packet arrival and add it to the tracking dict
        Args:
            obs (list): observation from the environment
        """
        self.count_new_pkts += 1
        Agent.total_new_rcv_pkts += 1
        Agent.total_data_size += self.pkt_size
        ## add to tracked pkts
        Agent.pkt_tracking_dict[int(self.pkt_id)]= {"src": self.index,
                                                    "node": self.index,
                                                    "dst": int(obs[0]),
                                                    "hops": [self.index],
                                                    "delays_ideal": [],
                                                    "delays_real": [],
                                                    "start_time": Agent.curr_time,
                                                    "tag": None}

    def handle_transit_packet(self, obs, done_flag):
        """ Handle a transit packet (not new). 
        Args:
            obs (list): observation from the environment
            done_flag (bool): if the packet arrived to destination
        """
        states_info = Agent.temp_obs.pop(self.pkt_id)

        hop_time_real =  Agent.curr_time - states_info["time"]
        hop_time_ideal = ((states_info["obs"][states_info["action"] + 1] + 512 ) * 8 / Agent.link_cap) + Agent.link_delay
        Agent.total_rewards_with_loss += hop_time_real
        ## add to tracked pkts
        Agent.pkt_tracking_dict[int(self.pkt_id)]["hops"].append(self.index)                        
        Agent.pkt_tracking_dict[int(self.pkt_id)]["node"] = self.index
        
        ## saving state info
        states_info["hop_time_real"] = hop_time_real
        states_info["hop_time_ideal"] = hop_time_ideal
        Agent.rewards.append(hop_time_real)
        Agent.pkt_tracking_dict[int(self.pkt_id)]["delays_ideal"].append(hop_time_ideal)
        Agent.pkt_tracking_dict[int(self.pkt_id)]["delays_real"].append(hop_time_real)                  
        
        if Agent.signaling_type == "ideal":
            Agent.replay_buffer[int(states_info["node"])].add(np.array(states_info["obs"], dtype=float).squeeze(),
                                                        states_info["action"], 
                                                        hop_time_real,
                                                        np.array(obs, dtype=float).squeeze(), 
                                                        done_flag)
        elif Agent.signaling_type == "NN":
            self._push_upcoming_event(int(states_info["node"]), { "time": Agent.curr_time + self.small_signaling_delay,
                                                                        "obs" : np.array(states_info["obs"], dtype=float).squeeze(),
                                                                        "action": states_info["action"], 
                                                                        "reward": hop_time_real,
                                                                        "new_obs": np.array(obs, dtype=float).squeeze(), 
                                                                        "flag": done_flag,
                                                                        "pkt_id": self.pkt_id,
                                                                        })
            if Agent.signalingSim == 0 and self.train:
                Agent.small_signaling_overhead_counter += self.small_signaling_pkt_size
                Agent.small_signaling_pkt_counter += 1

        elif Agent.signaling_type == "target":
            ## compute the target value
            filtered_index = np.where(np.array(list(Agent.G.neighbors(self.index)))!=int(states_info["node"]))[0] # filter the net interface from where the pkt comes 
            target = Agent.agents[self.index].get_target_value(np.array([hop_time_real]),
                                                                np.array([obs]),
                                                                np.array([done_flag]), 
                                                                filtered_index)
            # target = hop_time_ideal + Agent.gamma * (1- int(self.done_flag)) * tf.reduce_min(Agent.agents[self.index].q_network(np.array([self.obs], dtype=float)), 1)
            
            self._push_upcoming_event(int(states_info["node"]), {   "time": Agent.curr_time + self.small_signaling_delay,
                                                                    "obs" : np.array(states_info["obs"], dtype=float).squeeze(),
                                                                    "action": states_info["action"], 
                                                                    "target": target.numpy().item(),
                                                                    "new_obs": np.array(obs, dtype=float).squeeze(), 
                                                                    "flag": done_flag,
                                                                    "pkt_id": self.pkt_id,
                                                                    })
            if Agent.signalingSim == 0 and self.train:
                Agent.small_signaling_overhead_counter += self.small_signaling_pkt_size
                Agent.small_signaling_pkt_counter += 1
                
        elif Agent.signaling_type == "digital_twin":
            ## compute the target values
            targets = np.array([])
            dests = []
            for destination_ in range(Agent.numNodes):
                if destination_ == self.index or (Agent.d_t_send_all_destinations == 0 and destination_ != obs[0]):
                    continue
                if len(targets) ==0 :
                    targets = Agent.agents[self.index].q_network(np.array([[destination_] + list(obs[1:])], dtype=float)).numpy().reshape(1, -1)
                else:
                    targets = tf.concat([targets, Agent.agents[self.index].q_network(np.array([[destination_] + list(obs[1:])], dtype=float)).numpy()], axis=0).numpy().squeeze()
                dests.append(destination_)
                # targets = Agent.agents[self.index].q_network(np.array([obs], dtype=float))
            # target = hop_time_ideal + Agent.gamma * (1- int(self.done)) * tf.reduce_min(Agent.agents[self.index].q_network(np.array([obs], dtype=float)), 1)
            
            self._push_upcoming_event(int(states_info["node"]), {   "time": Agent.curr_time + self.small_signaling_delay,
                                                                    "obs" : np.array(states_info["obs"], dtype=float).squeeze(),
                                                                    "action": states_info["action"], 
                                                                    "targets": targets,
                                                                    "dests": dests,
                                                                    "reward": hop_time_real,
                                                                    "new_obs": np.array(obs, dtype=float).squeeze(), 
                                                                    "flag": done_flag,
                                                                    "pkt_id": self.pkt_id,
                                                                    })
            if Agent.signalingSim == 0 and self.train:
                Agent.small_signaling_overhead_counter += self.small_signaling_pkt_size
                Agent.small_signaling_pkt_counter += 1


    def handle_done(self):
        """ Handle the case when a packet arrives at the destination
        """
        self.count_arrived_packets += 1
        Agent.total_arrived_pkts += 1
        # Agent.total_e2e_delay += delay_time
        hops =  len(Agent.pkt_tracking_dict[int(self.pkt_id)]["hops"]) - 1
        Agent.total_hops += hops
        Agent.total_e2e_delay += self.delay_time
        Agent.delays_ideal.append(sum(Agent.pkt_tracking_dict[int(self.pkt_id)]["delays_ideal"]))
        Agent.delays_real.append(sum(Agent.pkt_tracking_dict[int(self.pkt_id)]["delays_real"]))
        Agent.delays.append(self.delay_time)
        Agent.nb_hops.append(hops)
        if(len(Agent.nb_hops)>50):
            Agent.nb_hops = Agent.nb_hops[-50:]
        if(len(Agent.delays)>50):
            Agent.delays = Agent.delays[-50:]
        Agent.pkt_tracking_dict.pop(int(self.pkt_id))
        

    def _push_upcoming_event(self, node, info):
        """Put the info in the upcoming event queue

        Args:
            node (int): node index
            info (dict): info to store as event
        """
        Agent.upcoming_events[node].append(info)
        Agent.upcoming_events[node].sort(key=operator.itemgetter("time"))
        
    def _get_upcoming_events(self):
        """
        Go through the upcoming event list and check whether of signaling pkt arrived, then add it to replay buffer if small signaling or update target if big signaling
        """
        if Agent.signaling_type == "NN":
            while len(Agent.upcoming_events[self.index]) > 0:
                if Agent.upcoming_events[self.index][0]["time"]> Agent.curr_time:
                    break
                element = Agent.upcoming_events[self.index].pop(0)
                if "obs" in element.keys():
                    ## treat small signaling 
                    Agent.replay_buffer[self.index].add(element["obs"],
                                                    element["action"], 
                                                    element["reward"],
                                                    element["new_obs"], 
                                                    element["flag"])
                    #print(self.index, Agent.replay_buffer[self.index].sample(1))
                else:
                    ## treat big signaling
                    # print("receive sync event to %s to %s at %s" % (self.index, element["neighbor_idx"], element["time"]))
                    self._sync_current(element["neighbor_idx"])
                    
        elif Agent.signaling_type == "target":
            while len(Agent.upcoming_events[self.index]) > 0:
                if Agent.upcoming_events[self.index][0]["time"]> Agent.curr_time:
                    break
                element = Agent.upcoming_events[self.index].pop(0)
                Agent.replay_buffer[self.index].add(element["obs"],
                                                element["action"], 
                                                element["target"],
                                                element["new_obs"], 
                                                element["flag"])
                            
        elif Agent.signaling_type == "digital_twin":
            while len(Agent.upcoming_events[self.index]) > 0:
                if Agent.upcoming_events[self.index][0]["time"]> Agent.curr_time:
                    break
                element = Agent.upcoming_events[self.index].pop(0)
                Agent.replay_buffer[self.index].add(element["obs"],
                                                    element["action"], 
                                                    element["reward"],
                                                    element["new_obs"], 
                                                    element["flag"])

    def _get_upcoming_events_real(self, signaling_pkt_id=None, ):
        """
        Go through the upcoming event list and check whether of signaling pkt arrived, then add it to replay buffer if small signaling or update target if big signaling
        """
        for idx, element in enumerate(Agent.upcoming_events[self.index]):
            if Agent.signaling_type == "NN":
                if element["pkt_id"] == signaling_pkt_id:
                    Agent.replay_buffer[self.index].add(element["obs"],
                                                        element["action"], 
                                                        element["reward"],
                                                        element["new_obs"], 
                                                        element["flag"])
                    Agent.upcoming_events[self.index].pop(idx)
                    return
                        
            elif Agent.signaling_type == "target":
                if element["pkt_id"] == signaling_pkt_id:
                    
                    if Agent.prioritizedReplayBuffer:
                        Agent.replay_buffer[self.index].add(element["obs"],
                                                            element["action"], 
                                                            element["target"],
                                                            element["new_obs"], 
                                                            element["flag"],
                                                            element["gradient_step"])
                        if element["gradient_step"] > Agent.replay_buffer[self.index].latest_gradient_step[element["action"]]:
                            Agent.replay_buffer[self.index].latest_gradient_step[element["action"]] = element["gradient_step"] 
                            Agent.replay_buffer[self.index].update_priorities(Agent.replay_buffer[self.index].neighbors_idx[element["action"]],
                                                                            element["action"])
                    else:
                        Agent.replay_buffer[self.index].add(element["obs"],
                                                            element["action"],
                                                            element["target"],
                                                            element["new_obs"],
                                                            element["flag"])
                    Agent.upcoming_events[self.index].pop(idx)
                    return
                
            elif Agent.signaling_type == "digital_twin":
                if element["pkt_id"] == signaling_pkt_id:
                    Agent.replay_buffer[self.index].add(element["obs"],
                                                        element["action"], 
                                                        element["reward"],
                                                        element["new_obs"], 
                                                        element["flag"])
                    for ix, destination_ in enumerate(element["dests"]):
                        Agent.agents[self.index].neighbors_d_t_database[element["action"]].add([destination_] + list(element["new_obs"][1:]),
                                                                                            element["targets"][ix],
                                                                                            element["time"],)
                    Agent.upcoming_events[self.index].pop(idx)
                    return
        raise ValueError("signaling pkt id not found")