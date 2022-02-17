#!/usr/bin/env python3
# -*- coding: utf-8 -*-


### imports
import argparse
from time import sleep
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

### define the global params
# params overwritten by the args parser
startSim = 0
iterationNum = 100000
simTime = 20
stepTime = 0.01
seed = 0
simArgs = {"--simTime": simTime,
        "--testArg": 123}
debug = False
basePort = 5555
# params for the network topology
numNodes = 5
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
    print("index :", index,  port)
    
    ### declare the ns3 env
    env = ns3env.Ns3Env(port=int(port), stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
    obs = env.reset()
    print("*"*5, index, obs)
    envs[str(index)] = copy.deepcopy(env)
    
    ### declare the DQN buffer model
    # ob_space = env.observation_space
    # ac_space = env.action_space
    # print("index :", index, "Observation space: ", ob_space.shape,  ob_space.dtype)
    # print("index :", index, "Action space: ", ac_space, ac_space.n, ac_space.dtype)
    agents[index] = DQN_AGENT(
        q_func=DQN_buffer_model,
        observation_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        num_nodes=numNodes,
        input_size_splits = [1,
                             env.action_space.n +1,
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
        
    try:
        while True:
            obs = env.reset()
            while True:
                if(not env.connected):
                    break
                stepIdx += 1
                
                #Select the agent: Dijkstra / Random Agent
                #action = g.getInterface(obs[0])
                # action = env.action_space.sample()
                
                # schedule the exploration
                update_eps = tf.constant(exploration.value(currIt))
                
                # take the action using the model
                action = agents[index].step(obs, True, update_eps)
                
                # apply the action
                obs, reward, done, info = env.step(action)

                # store the observations and reward
                
                
                # print_lock.acquire()
                print("index :", index, "---obs, reward, done, info: ", obs, reward, done, info)
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
                print("index :", index, "Done by iterations number")
                break

    except KeyboardInterrupt:
        print("index :", index, "Ctrl-C -> Exit")
        env.close()
    finally:
        print("index :", index, "Curr Iter: ", currIt)
        avg_queue_size = np.array(avg_queue_size)
        avg_rew = np.array(avg_rew)
        avg_delay_time = np.array(avg_delay_time)
    print("index :", index, "Average Queue size")
    print("index :", index, "Max: ",avg_queue_size.max(axis=0), "Min: ",avg_queue_size.min(axis=0), "Mean: ", avg_queue_size.mean(axis=0), "Std: ", avg_queue_size.std(axis=0))
    print("index :", index, "Max: ",avg_queue_size.max(), "Min: ",avg_queue_size.min(), "Mean: ", avg_queue_size.mean(), "Std: ", avg_queue_size.std())
    print("index :", index, "-------------------------------------------------")

    print("index :", index, "Average Reward Time")
    print("index :", index, "Max: ",avg_rew.max(), "Min: ",avg_rew.min(), "Mean: ", avg_rew.mean(), "Std: ", avg_rew.std())
    print("index :", index, "-------------------------------------------------")

    print("index :", index, "Average Delay Time")
    print("index :", index, "QTD: ",avg_delay_time.shape)
    print("index :", index, "Max: ",avg_delay_time.max(), "Min: ",avg_delay_time.min(), "Mean: ", avg_delay_time.mean(), "Std: ", avg_delay_time.std())
    print("index :", index, "-------------------------------------------------")
    print("index :", index, "Recv Packets: ", count_recv_packets)
    print("index :", index, "Loss Packet rate: ", max(0, float(1-count_recv_packets/count_packets_sent)))

    env.close()
    print("index :", index, "Done")


def main():
    ### retrieve the params from script args
    parser = argparse.ArgumentParser(description='Start simulation script on/off')
    parser.add_argument('--start',
                        type=int,
                        default=1,
                        help='Start ns-3 simulation script 0/1, Default: 1')
    parser.add_argument('--iterations',
                        type=int,
                        default=10,
                        help='Total number of iterations, Default: ---')
    parser.add_argument('--port',
                        type=int,
                        default=5555,
                        help='Base Port number, Default: 5555')
    parser.add_argument('--simTime',
                        type=int,
                        default=20,
                        help='simTime in seconds, Default: 20')
    parser.add_argument('--stepTime',
                        type=int,
                        default=0.01,
                        help='stepTime in seconds, Default: 0.01')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='seed, Default: 0')
    args = parser.parse_args()
    startSim = bool(args.start)
    iterationNum = int(args.iterations)
    basePort = int(args.port)
    simTime = int(args.simTime) # seconds
    stepTime = float(args.stepTime)  # seconds
    seed = int(args.seed)
    simArgs = {"--simTime": simTime,
            "--testArg": 123}
    debug = False
    
    ### fix the seed
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



    
    # run threads
    threads = []
    for index in range(numNodes):
        x = threading.Thread(target=agent_action, args=(index,))
        threads.append(x)
        x.start()
        print(f"index : {index}, identifier : {x}")
    sleep(1)
    while threading.active_count() > 1:
        print(f"threading count : {threading.active_count()}, envs : {envs}")    
        sleep(100)


if __name__ == '__main__':
    main()
   