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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


### define the global params
# params overwritten by the args parser
startSim = 0
iterationNum = 1000
simTime = 200
stepTime = 0.01
seed = 0
simArgs = {"--simTime": simTime,
        "--testArg": 123}
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
# replay buffer max size
replay_buffer_max_size = 5000
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

### define the agent fonction
def agent_action(index):
    """Define the agent action that will be threaded.
    The function will reset the environment, and then retrieve observation to take the proper actions.
    It uses the global variables to store it states, action, reward flags tuples.

    Args:
        index (int, optional): the agent index. Defaults to 0.
    """
    
    ### compute the port number
    port = basePort + index
    print("index :", index,  port, startSim)
    
    ### declare the ns3 env
    # sleep(np.random.random()*5)
    env = ns3env.Ns3Env(port=int(port), stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
    obs = env.reset()
    # print(env, out=logfile)
    # print("*"*5, index, obs, out=logfile)
    envs[str(index)] = env
    
    ### declare the DQN buffer model
    # ob_space = env.observation_space
    # ac_space = env.action_space
    # print("index :", index, "Observation space: ", ob_space.shape,  ob_space.dtype, out=logfile)
    # print("index :", index, "Action space: ", ac_space, ac_space.n, ac_space.dtype, out=logfile)
    print("index :", index, "Action space: ", env.action_space.n,"Observation space: ", (env.action_space.n+1,), "split",[1,env.action_space.n,], obs)
    agents[index] = DQN_AGENT(
        q_func=DQN_buffer_model,
        observation_shape=(env.action_space.n+1,),
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
    avg_queue_size = []
    avg_rew = []
    avg_delay_time = []
    avg_packet_size = []
    count_recv_packets = 0
    stepIdx = 0
    currIt = 0
    count_packets_sent = 0
    logfile = open(f'outputs/out_{port}.txt', "w")
    try:
        while True:
            obs = env.reset()
            # retrieve pkt id and obs
            pkt_id = obs.pop()
            while True:
                if(not env.connected):
                    break
                stepIdx += 1
                
                # schedule the exploration
                update_eps = tf.constant(exploration.value(currIt))
                
                # take the action using the model
                # action = env.action_space.sample()
                print([obs], file=logfile)
                if obs[0] == index: # pkt arrived to dst
                    action = 0
                else:
                    action = np.argmin(agents[index].step(np.array([obs]), True, update_eps))
                    # add to the temp obs 
                    temp_obs[pkt_id]= {"node": index, "obs": obs, "action": action}
                    
                # apply the action
                obs, reward, done, info = env.step(action)
                
                # retrieve pkt id and obs
                pkt_id = obs.pop()
                
                # store the observations and reward
                if pkt_id in temp_obs.keys(): ## check if the packet is not new in the network
                    states_info = temp_obs.pop(pkt_id)
                    replay_buffer[int(states_info["node"])].add(np.array(states_info["obs"], dtype=float).squeeze(),
                                                                states_info["action"], 
                                                                reward,
                                                                np.array(obs, dtype=float).squeeze(), 
                                                                done)
                    print(f"adds element {pkt_id} to replay buffer of node {states_info}", file=logfile)
                else:
                    temp_obs[pkt_id]= {"node": index, "obs": obs, "action": action}

                print("index :", index, "---obs, reward, done, info: ", obs, reward, done, info, file=logfile)
                print(temp_obs, file=logfile)
                tokens = info.split(",")
                delay_time = float(tokens[0].split('=')[-1])
                count_packets_sent = int(tokens[1].split('=')[-1])
                avg_rew.append(reward)
                avg_queue_size.append(obs[1:])

                if done:
                    if(delay_time<50000):
                        avg_delay_time.append(delay_time)
                        count_recv_packets += 1
                    stepIdx = 0
                    if currIt + 1 < iterationNum:
                        env.reset()
                        pass
                    break
                
            if(not env.connected):
                break
            currIt += 1
            if currIt == iterationNum:
                print("index :", index, "Done by iterations number", file=logfile)
                break

    except KeyboardInterrupt:
        print("index :", index, "Ctrl-C -> Exit", file=logfile)
        env.close()
    finally:
        print("index :", index, "Curr Iter: ", currIt, file=logfile)
        avg_queue_size = np.array(avg_queue_size)
        avg_rew = np.array(avg_rew)
        avg_delay_time = np.array(avg_delay_time)
    if len(avg_queue_size):
        print("index :", index, "Average Queue size", file=logfile)
        print("index :", index, "Max: ",avg_queue_size.max(axis=0), "Min: ",avg_queue_size.min(axis=0), "Mean: ", avg_queue_size.mean(axis=0), "Std: ", avg_queue_size.std(axis=0), file=logfile)
        print("index :", index, "Max: ",avg_queue_size.max(), "Min: ",avg_queue_size.min(), "Mean: ", avg_queue_size.mean(), "Std: ", avg_queue_size.std(), file=logfile)
        print("index :", index, "-------------------------------------------------", file=logfile)
    if len(avg_rew):
        print("index :", index, "Average Reward Time", file=logfile)
        print("index :", index, "Max: ",avg_rew.max(), "Min: ",avg_rew.min(), "Mean: ", avg_rew.mean(), "Std: ", avg_rew.std(), file=logfile)
        print("index :", index, "-------------------------------------------------", file=logfile)
    if len(avg_delay_time):
        print("index :", index, "Average Delay Time", file=logfile)
        print("index :", index, "QTD: ",avg_delay_time.shape, file=logfile)
        print("index :", index, "Max: ",avg_delay_time.max(), "Min: ",avg_delay_time.min(), "Mean: ", avg_delay_time.mean(), "Std: ", avg_delay_time.std(), file=logfile)
        print("index :", index, "-------------------------------------------------", file=logfile)
    print("index :", index, "Recv Packets: ", count_recv_packets, file=logfile)
    print("index :", index, "Loss Packet rate: ", max(0, float(1-count_recv_packets/count_packets_sent)), file=logfile)

    env.close()
    print("index :", index, "Done", file=logfile)


def main():
    
    global currIt, stepIdx
    stepIdx = 0
    currIt = 0

    ### fix the seed
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    
    # run threads
    global threads
    threads = []
    for index in range(numNodes):
        x = threading.Thread(target=agent_action, args=(index,))
        threads.append(x)
    for thread in threads:
        thread.start()
    # print(f"index : {index}, identifier : {x}", out=logfile)
        
    sleep(5)
    while threading.active_count() > 1:
        # print(f"threading count : {threading.active_count()}, envs : {envs}")
        sleep(1)


if __name__ == '__main__':
    a = time()
    main()
    print(time() - a)