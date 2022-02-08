#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from ns3gym import ns3env
from numpy.lib.function_base import append
from ns3gym.graph import *
import numpy as np

#from scratch.my_network.ns3gym.graph import Graph

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2018, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "gawlowicz@tkn.tu-berlin.de"


parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start',
                    type=int,
                    default=1,
                    help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations',
                    type=int,
                    default=10000000,
                    help='Number of iterations, Default: ---')
parser.add_argument('--port',
                    type=int,
                    default=1,
                    help='Port, Default: 1')
parser.add_argument('--index',
                    type=int,
                    default=0,
                    help='Index, Default: 0')


args = parser.parse_args()
startSim = bool(args.start)
iterationNum = int(args.iterations)
port = int(args.port)
index = int(args.index)



env = ns3env.Ns3Env(port=port, startSim=startSim)
env.reset()

g = Graph(5, index)
g.openFile()
g.dijkstra()


ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)

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
            action = env.action_space.sample()
            
            
            obs, reward, done, info = env.step(action)
            #print("---obs, reward, done, info: ", obs, reward, done, info)
            tokens = info.split(",")
            delay_time = float(tokens[0].split('=')[-1])
            packets_sent = float(tokens[1].split('=')[-1])
            avg_rew.append(reward)
            avg_queue_size.append(obs[2:])

           


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
            print("Done by iterations number")
            break

except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    print("Curr Iter: ", currIt)
    avg_queue_size = np.array(avg_queue_size)
    avg_rew = np.array(avg_rew)
    avg_delay_time = np.array(avg_delay_time)


    #Printing some metrics 
   
    
    print("Average Queue size")
    print("Max: ",avg_queue_size.max(axis=0), "Min: ",avg_queue_size.min(axis=0), "Mean: ", avg_queue_size.mean(axis=0), "Std: ", avg_queue_size.std(axis=0))
    print("Max: ",avg_queue_size.max(), "Min: ",avg_queue_size.min(), "Mean: ", avg_queue_size.mean(), "Std: ", avg_queue_size.std())
    print("-------------------------------------------------")
    
    print("Average Reward Time")
    print("Max: ",avg_rew.max(), "Min: ",avg_rew.min(), "Mean: ", avg_rew.mean(), "Std: ", avg_rew.std())
    print("-------------------------------------------------")
    
    print("Average Delay Time")
    print("QTD: ",avg_delay_time.shape)
    print("Max: ",avg_delay_time.max(), "Min: ",avg_delay_time.min(), "Mean: ", avg_delay_time.mean(), "Std: ", avg_delay_time.std())
    print("-------------------------------------------------")

    print("Recv Packets: ", count_recv_packets)
    #print("Src Nodes: ", count_recv_nodes)

    env.close()
    print("Done")