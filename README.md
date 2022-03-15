Network Simulation for Reinforcement Learning (NetSim)
============

The NetSim is a framework developed for Reinformecent Learning (RL) application for packet routing. This framework is based on the OpenAI Gym toolkit and the Ns3 library.

The [OpenAI Gym](https://gym.openai.com/) is a toolkit for RL widely used in research. The network simulator [ns–3](https://www.nsnam.org/) is a standard library, which may provide useful simulation tools. It generates discrete events and provide several protocol implementations.

Moreover, the NetSim implementation is based on [ns3-gym](https://github.com/tkn-tub/ns3-gym), which integrates OpenAI Gym and ns-3.

Installation
============

1. Install all required dependencies required by ns-3.
```
# minimal requirements for C++:
apt-get install gcc g++ python

see https://www.nsnam.org/wiki/Installation
```
2. Install ZMQ and Protocol Buffers libs:
```
# to install protobuf-3.6 on ubuntu 16.04:
sudo add-apt-repository ppa:maarten-fonville/protobuf
sudo apt-get update

apt-get install libzmq5 libzmq5-dev
apt-get install libprotobuf-dev
apt-get install protobuf-compiler
```

3. Install ns3gym located in src/opengym/model/ns3gym (Python3 required)
```
pip3 install ./src/opengym/model/ns3gym
```

5. Install Tensorflow
```
pip install --upgrade tensorflow
```

6. Run agent and simulation:
```
cd ./my_network/
./launch_dqn.sh
```

7. (Optional) For killing agents, use the script:
```
./kill_agents.sh
```

Agents
===========

The netsim provides 2 types of agents: Shortest Path (SP) and Deep Q-Network (DQN).

## DQN 

For using the DQN agent to train, uncomment the first block of the file launch_dqn.sh . 
For testing, uncomment the 2nd block of the script launch_dqn.sh . 


##  SP

For using the Shortest Path algorithm, uncomment the block of lines in the launch_sp.sh file corresponding to this file.

