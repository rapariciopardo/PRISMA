Network Simulation for Reinforcement Learning (NetSim)
============

The NetSim is a framework developed for Reinformecent Learning (RL) application for packet routing. This framework is based on the OpenAI Gym toolkit and the Ns3 library.

The [OpenAI Gym](https://gym.openai.com/) is a toolkit for RL widely used in research. The network simulator [ns–3](https://www.nsnam.org/) is a standard library, which may provide useful simulation tools. It generates discrete events and provide several protocol implementations.

Moreover, the NetSim implementation is based on [ns3-gym](https://github.com/tkn-tub/ns3-gym), which integrates OpenAI Gym and ns-3.

Installation
============

If you don't have the Nse-gym alrady installed

1. If the submodule was not yet initialized, initialize them.
```
git submodule init
git submodule update
```

2. Run the script install.sh . It will install the NS3 requirements (minimal requirements for C++, ZMQ and Protocol Buffers libs. For more information, see https://www.nsnam.org/wiki/Installation). Moreover, it will compile the messages.proto file for python. 

The usage of sudo may be required.
```
sudo sh install.sh
```

3. Go to my_network directory and install the python packages required using the command below (numpy, networkx, gym, tensorflow, zmq)
```
cd ./my_network/
pip install -e .
```

4. For running Q-Routing Run agent and simulation:
```
./launch_q_routing.sh
```
or

For launching Shortest Path agent:
```
./launch_sp.sh
```


5. (Optional) For killing agents, use the script:
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

