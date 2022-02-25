#!/usr/bin/env python3
# -*- coding: utf-8 -*-


### imports
import argparse
from time import sleep, time
from ns3gym import ns3env
from ns3gym.graph import *
import numpy as np
import threading
import copy
from dqn_agent.learner import DQN_AGENT
from dqn_agent.replay_buffer import ReplayBuffer
from dqn_agent.models import DQN_buffer_model
from dqn_agent.utils import save_model, load_model, LinearSchedule
import tensorflow as tf
import random
import networkx as nx
import os
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


### define the global trackers 
currIt = 0
total_new_rcv_pkts = 0
total_arrived_pkts = 0
total_rewards = 0
total_lost_pkts = 0
# params overwritten by the args parser
startSim = 0
iterationNum = 1000
simTime = 200
stepTime = 0.01
seed = 0
simArgs = {"--simTime": simTime,
        "--testArg": 123}

max_buffer_size = 100000
link_cap = 10000000
loss_penalty = (max_buffer_size*8)/link_cap

debug = False
basePort = 6555
# params for the network topology
numNodes = 11
# network topology
G=nx.Graph()
for i, element in enumerate(np.loadtxt(open("node_coordinates.txt"))):
    G.add_node(i,pos=tuple(element))
G = nx.from_numpy_matrix(np.loadtxt(open("adjacency_matrix.txt")), create_using=G)
# params for training
lr = 0.01
gamma = 0.9
batch_size = 128
target_sync_step = 64
between_train_steps_delay = 0.05
# replay buffer max size
replay_buffer_max_size = 10000
# exploration ratio 
exploration_fraction = 1
exploration_initial_eps = 0.5
exploration_final_eps = 0.1
# dict storing the params
envs = {}
# define the agents
agents = [None]*numNodes
# define the replay buffer as a global variable
replay_buffer = [ReplayBuffer(replay_buffer_max_size) for n in range(numNodes)]
# define the temp observations dict to prepare the (s, a, r, s', flag) for replay buffer
temp_obs = {}
# # Simulation trackers
# currIt = 0


### define the agent fonction
def agent_action(index):
    """Define the agent action that will be threaded.
    The function will reset the environment, and then retrieve observation to take the proper actions.
    It uses the global variables to store it states, action, reward flags tuples.

    Args:
        index (int, optional): the agent index. Defaults to 0.
    """
    
    ### compute the port number
    global currIt, total_new_rcv_pkts, total_arrived_pkts, total_rewards, total_lost_pkts
    port = basePort + index
    print("index :", index,  port, startSim)
    
    ### decalre node neighbors
    neighbors = list(G.neighbors(index))

    ### declare the ns3 env
    env = ns3env.Ns3Env(port=int(port), stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
    obs = env.reset()

    envs[str(index)] = env
    
    ### declare the DQN buffer model
    # ob_space = env.observation_space
    # ac_space = env.action_space
    # print("index :", index, "Observation space: ", ob_space.shape,  ob_space.dtype, out=logfile)
    # print("index :", index, "Action space: ", ac_space, ac_space.n, ac_space.dtype, out=logfile)
    # print("index :", index, "Action space: ", env.action_space.n,"Observation space: ", (env.action_space.n+1,), "split",[1,env.action_space.n,], obs)
    agents[index] = DQN_AGENT(
        q_func=DQN_buffer_model,
        observation_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        num_nodes=numNodes,
        input_size_splits = [1,
                             env.action_space.n,
                             ],
        lr=lr,
        gamma=gamma
    )

    # Create the schedule for exploration.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * iterationNum),
                                 initial_p=exploration_initial_eps,
                                 final_p=exploration_final_eps)
    # avg_queue_size = []
    # avg_rew = []
    # avg_delay_time = []
    # avg_packet_size = []
    episode_mean_td_error = []
    count_arrived_packets = 0
    count_new_pkts = 0
    last_training_time = 0
    # currIt = 0
    # count_packets_sent = 0
    gradient_step_idx = 0
    logfile = open(f'outputs/out_{port}.txt', "w")
    try:
        while True:
            obs = env.reset()
            stepIdx = 0
            # print(index, obs, "definition")
            while True:
                if(not env.connected):
                    break
                
                ### schedule the exploration
                update_eps = tf.constant(exploration.value(stepIdx))
                
                ### take the action
                # action = env.action_space.sample()       
                if obs[0] == index or stepIdx < 1: # pkt arrived to dst or it is a train step, ignore the action
                    action = 0
                else:
                    ### Take action using the NN
                    action = agents[index].step(np.array([obs]), True, update_eps).numpy().item()
                    # print(action)*
                    if obs[action + 1] > max_buffer_size:
                        replay_buffer[index].add(np.array(obs, dtype=float).squeeze(),
                                            action, 
                                            loss_penalty,
                                            np.array(np.ones(list(G.neighbors(neighbors[action]))), dtype=float).squeeze(), 
                                            True)
                        total_lost_pkts += 1
                    ### Add to the temp obs
                    else: 
                        temp_obs[pkt_id]= {"node": index, "obs": obs, "action": action, "time": curr_time}

                ### Apply the action
                obs, reward, done, info = env.step(action)
                
                # print(obs)
                ###check if episode is done
                if done and obs[0] == -1:
                    # print("episode done")
                    break
                
                ### Increment the simulation and episode counters
                stepIdx += 1
                currIt += 1

                ### info treatments
                tokens = info.split(",")
                delay_time = float(tokens[0].split('=')[-1])
                # count_packets_sent = int(tokens[1].split('=')[-1])
                curr_time = float(tokens[4].split('=')[-1])
                pkt_id = float(tokens[5].split('=')[-1])
                
                if done:
                    if(delay_time<50000):
                        count_arrived_packets += 1
                        total_arrived_pkts += 1
                # print("index :", index, "---obs, reward, done, info;, temp_obs ,action, tokens: ", obs, reward, done, info, temp_obs, action, tokens)

                # store the observations and reward
                if pkt_id in temp_obs.keys(): ## check if the packet is not new in the network
                    states_info = temp_obs.pop(pkt_id)
                    hop_time =  curr_time - states_info["time"]
                    replay_buffer[int(states_info["node"])].add(np.array(states_info["obs"], dtype=float).squeeze(),
                                                                states_info["action"], 
                                                                curr_time - states_info["time"],
                                                                np.array(obs, dtype=float).squeeze(), 
                                                                done)
                    total_rewards += curr_time - states_info["time"]
                else:
                    count_new_pkts += 1
                    total_new_rcv_pkts += 1

                ### Do a gradient Step
                if curr_time > (last_training_time + between_train_steps_delay) and len(replay_buffer[index])> batch_size:
                    ### Sync target NN
                    for indx, neighbor in enumerate(neighbors): 
                        agents[index].sync_neighbor_target_q_network(agents[neighbor].q_network, indx)

                    # print("train...", index)
                    obses_t, actions_t, rewards_t, next_obses_t, dones_t = replay_buffer[index].sample(batch_size)
                    weights, _ = np.ones(batch_size, dtype=np.float32), None

                    ### Construct the target values
                    targets_t = []
                    action_indices_all = []
                    for indx, neighbor in enumerate(neighbors):
                        filtered_indices = np.where(np.array(list(G.neighbors(neighbor)))!=index)[0] # filter the net interface from where the pkt comes
                        action_indices = np.where(actions_t == indx)[0]
                        action_indices_all.append(action_indices)
                        if len(action_indices):
                            targets_t.append(agents[index].get_neighbor_target_value(indx, rewards_t[action_indices], tf.constant(
                                np.array(np.vstack(next_obses_t[action_indices]), dtype=np.float)), dones_t[action_indices], filtered_indices))
                    action_indices_all = np.concatenate(action_indices_all)

                    ### prepare tf variables
                    obses_t = tf.constant(obses_t[action_indices_all,])
                    actions_t = tf.constant(actions_t[action_indices_all], shape=(batch_size))
                    targets_t = tf.constant(tf.concat(targets_t, axis=0), shape=(batch_size))
                    weights = tf.constant(weights)

                    ### Make a gradient step
                    td_errors = agents[index].train(obses_t, actions_t, targets_t, weights)    
                    episode_mean_td_error.append(np.mean(td_errors))
                    # print(index, "td error", np.mean(td_errors))
                    gradient_step_idx += 1

                    # stepIdx = 0
                    # if currIt + 1 < iterationNum:
                    #     env.reset()
                    #     pass
                    # break
            # print("episode ends", index)
            # env.close()
            break

            if(not env.connected):
                break
            # currIt += 1
            if currIt == iterationNum:
                print("index :", index, "Done by iterations number", file=logfile)
                env.close()
                break

    except KeyboardInterrupt:
        print("index :", index, "Ctrl-C -> Exit", file=logfile)
        env.close()
        
    # finally:

    #     print("index :", index, "Curr Iter: ", stepIdx, file=logfile)
    #     avg_queue_size = np.array(avg_queue_size)
    #     avg_rew = np.array(avg_rew)
    #     avg_delay_time = np.array(avg_delay_time)
    # if len(avg_queue_size):
    #     print("index :", index, "Average Queue size", file=logfile)
    #     print("index :", index, "Max: ",avg_queue_size.max(axis=0), "Min: ",avg_queue_size.min(axis=0), "Mean: ", avg_queue_size.mean(axis=0), "Std: ", avg_queue_size.std(axis=0), file=logfile)
    #     print("index :", index, "Max: ",avg_queue_size.max(), "Min: ",avg_queue_size.min(), "Mean: ", avg_queue_size.mean(), "Std: ", avg_queue_size.std(), file=logfile)
    #     print("index :", index, "-------------------------------------------------", file=logfile)
    # if len(avg_rew):
    #     print("index :", index, "Average Reward Time", file=logfile)
    #     print("index :", index, "Max: ",avg_rew.max(), "Min: ",avg_rew.min(), "Mean: ", avg_rew.mean(), "Std: ", avg_rew.std(), file=logfile)
    #     print("index :", index, "-------------------------------------------------", file=logfile)
    # if len(avg_delay_time):
    #     print("index :", index, "Average Delay Time", file=logfile)
    #     print("index :", index, "QTD: ",avg_delay_time.shape, file=logfile)
    #     print("index :", index, "Max: ",avg_delay_time.max(), "Min: ",avg_delay_time.min(), "Mean: ", avg_delay_time.mean(), "Std: ", avg_delay_time.std(), file=logfile)
    #     print("index :", index, "-------------------------------------------------", file=logfile)
    # print("index :", index, "Recv Packets: ", count_arrived_packets, file=logfile)
    # print("index :", index, "Loss Packet rate: ", max(0, float(1-count_arrived_packets/count_packets_sent)), file=logfile)
    # env.close()

    env.ns3ZmqBridge.send_close_command()
    print("index :", index, "Done", "stepIdx =", stepIdx, "arrived pkts =", count_arrived_packets,  "new received pkts", count_new_pkts, episode_mean_td_error)
    return True

def main():
    ### fix the seed
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # run threads
    # global threads
    # threads = []
    for index in range(numNodes):
        x = threading.Thread(target=agent_action, args=(index,))
        x.start()
        # threads.append(x)
    # print(f"index : {index}, identifier : {x}", out=logfile)
        
    sleep(1)
    while threading.active_count() > 1:
        # print(f"threading count : {threading.active_count()}")
        sleep(1)
    print(f""" Summary of the episode :
            Total number of Transitions = {currIt}, 
            Total e2e delay = {total_rewards}, 
            Total number of packets = {total_new_rcv_pkts}, 
            Number of arrived packets = {total_arrived_pkts},
            Number of lost packets = {total_lost_pkts},
            Loss ratio = {total_lost_pkts/total_new_rcv_pkts},
            """)
    if total_arrived_pkts:
        print(f"Average delay per arrived packets = {total_rewards/total_arrived_pkts}")

if __name__ == '__main__':

    start_time = time()
    main()
    print("Elapsed time = ", str(datetime.timedelta(seconds= time() - start_time)))