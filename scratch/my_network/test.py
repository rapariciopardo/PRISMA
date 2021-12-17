#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from ns3gym import ns3env
from ns3gym.graph import *

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
                    default=10,
                    help='Number of iterations, Default: 1')
parser.add_argument('--port',
                    type=int,
                    default=1,
                    help='Port, Default: 1')
parser.add_argument('--index',
                    type=int,
                    default=0,
                    help='Index, Default: 0')

file = open("DoneAll.txt","w")
file.write("False")
file.close()

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

stepIdx = 0
currIt = 0



try:
    while True:
        print("Start iteration: ", currIt)
        obs = env.reset()
        print("Step: ", stepIdx)
        print("---obs: ", obs)

        while True:
            file = open("DoneAll.txt","r")
            if(file.read()=='True'):
                break
            file.close()
            stepIdx += 1
            action = g.getInterface(obs[0]) #env.action_space.sample()
            print("---action: ", action)

            print("Step: ", stepIdx)
            obs, reward, done, info = env.step(action)
            print("---obs, reward, done, info: ", obs, reward, done, info)

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
        file = open("DoneAll.txt","r")
        if(file.read()=='True'):
            print("aqui... Break")
            file2 = open(str(index)+".txt","w")
            file2.write(str(index))
            file2.close()
            break
        file.close()
        currIt += 1
        if currIt == iterationNum:
            break

except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    file = open("DoneAll.txt","w")
    file.write("True")
    file.close()
    env.close()
    print("Done")