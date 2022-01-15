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

#port = 5555
simTime = 20 # seconds
stepTime = 0.01  # seconds
seed = 0
simArgs = {"--simTime": simTime,
           "--testArg": 123}
debug = False



env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
g = Graph(5, index)
g.openFile()
g.dijkstra()
#g.getRoutingTable()
#print("Rounting Table: ", g.RoutingTable)
# simpler:
#env = ns3env.Ns3Env()
env.reset()

ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)

avg_queue_size = []
avg_rew_time = []
stepIdx = 0
currIt = 0
counter = 0



try:
    while True:
        #print("Start iteration: ", currIt)
        obs = env.reset()
        #print("Step: ", stepIdx)
        #print("---obs: ", obs)

        while True:
            if(not env.connected):
                break
            stepIdx += 1
            
            #action = g.getInterface(obs[0]) #env.action_space.sample()
            action = env.action_space.sample()
            #print("---action: ", action)

            #print("Step: ", stepIdx)
            obs, reward, done, info = env.step(action)
            #print("---obs, reward, done, info: ", obs, reward, done, info)
            avg_rew_time.append(reward)
            avg_queue_size.append(obs[2:])

            #if(stepIdx==2):
            #    import global_
            #    global_.doneAll = True

            
            #print("Done All: ", doneAll)


            if done:
                stepIdx = 0
                if currIt + 1 < iterationNum:
                    env.reset()
                    pass
                break
            else:
                counter += 1
        if(not env.connected):
            break
        currIt += 1
        if currIt == iterationNum:
            print("Done mesmo...")
            break

except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    print("Curr Iter: ", currIt)
    avg_queue_size = np.array(avg_queue_size)
    avg_rew_time = np.array(avg_rew_time)
    print(avg_queue_size)
    print("Average Queue")
    print("Max: ",avg_queue_size.max(axis=0), "Min: ",avg_queue_size.min(axis=0), "Mean: ", avg_queue_size.mean(axis=0), "Std: ", avg_queue_size.std(axis=0))
    print("Max: ",avg_queue_size.max(), "Min: ",avg_queue_size.min(), "Mean: ", avg_queue_size.mean(), "Std: ", avg_queue_size.std())

    print("-------------------------------------------------")
    print(avg_rew_time)
    print("Average Reward Time")
    print("Max: ",avg_rew_time.max(), "Min: ",avg_rew_time.min(), "Mean: ", avg_rew_time.mean(), "Std: ", avg_rew_time.std())
    print("-------------------------------------------------")
    print("Counter: ", counter)

    env.close()
    print("Done")