PRISMA-v2: A Packet Routing Simulator for Multi-Agent Reinforcement Learning - Extension to CLoud Overlay Networks
============

PRISMA-v2 is a network simulation playground for developing and testing Multi-Agent Reinforcement Learning (MARL) solutions for dynamic packet routing (DPR). This framework is based on the OpenAI Gym toolkit and the Ns3 simulator.

The [OpenAI Gym](https://gym.openai.com/) is a toolkit for RL widely used in research. The network simulator [ns–3](https://www.nsnam.org/) is a standard library, which may provide useful simulation tools. It generates discrete events and provide several protocol implementations.

Overlay networks are virtual networks built on top of a physical network (called underlay networks) utilizing network virtualization technology. Overlay networks provide flexible and dynamic traffic routing between nodes that are not directly connected by physical links, but rather by virtual links that correspond to paths in the underlying network.The PRISMA-v2 is the extension of PRISMA framework and allows the developing of MARL for DPR in overlay networks.

The main contributions of this framework:
1) Overlay topology simulation and control management.
2) Ability to add dynamic underlay traffic along with the overlay one.
3) High reproducibility of results by supplying containerizing capability using docker [Docker](https://www.docker.com/).
4) Refactoring the code for better readability.
5) Improve Tensorboard logging by incorporating both training and testing phases.
6) Implement control packets to realistically simulate the communication between the nodes and evaluate the overhead of running a DRL approach.

Installation
============
I- Local Installation
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
cd ./prisma/
pip install -e .
```

4. For training or testing, run the main program with the parameters:
```
python3 main.py $PARAMS
```
II- Using Docker
============
## 1. Build the docker image locally

a) Install docker dependencies

b) build the docker image using the docker file
```
docker build -t prisma .
```
c) run the docker image and bind only the examples folder to the container
```
sudo docker run --rm --gpus all -v ./prisma/examples:/app/prisma/examples -w /app/prisma prisma /bin/bash -c "python3 main.py $PARAMS"
```
## 2. Use an existing environment from DockerHub
a) Pull the docker image from dockerhub
```
docker pull allicheredha/prisma_env
```
b) run the docker image and bind the complete folder to the container
```
sudo docker run --rm --gpus all -v ./prisma/:/prisma -w /prisma allicheredha/prisma_env /bin/bash -c "python3 main.py $PARAMS"
```

Usage guide
===========

For training the agent, run the script train.sh. In order to test a model, run the script test.sh.  

In the script, you can change several parameters. The parameters are divided into 4 categories. The main parameters are the following:
## Global Simulation
```
numEpisodes: Number of episodes.
simTime: Simulation time (in seconds) of one episode.
basePort: Base TCP Port for agent communication. 
seed: Seed for simulation.
train: 1, if training; 0, Otherwise.
```
## Underlay Network
```
load_factor: Defines a factor multiplied by traffic rate matrix 
physical_adjacency_matrix_path: Path for adjacency matrix
traffic_matrix_path: Path for traffic rate matrix
max_out_buffer_size: maximum size of the output buffers (in bytes)
link_delay: Defines the delay of the link
packet_size: Defines the packet size.
link_cap: Defines the rate a packet is uploaded to the link.
```
## Overlay Network
```
overlay_adjacency_matrix_path: Path for overlay adjacency matrix
map_overlay_path: Path for file which maps undelar and overlay node.
indexes
pingAsObs: If true, we use the tunnel delay info (recovered by the ping packets) as observation state.
pingPacketIntervalTime: The period between two ping packets (active when pingAsObs is true).

```
## Agent
```
lr: Learning rate used for training
agent_type: "dqn", "dq_routing", "sp" (Shostest Path), "opt" (oracle policy routing)
gamma: Discount factor $\gamma$. 
load_path: Path to load a model (optional)
training_step: Defines the step in training in secs.
replay_buffer_max_size: Maximum replay buffer size.
snapshot_interval: Defines the interval between two snapshots model saving.
```
## Session logging 
```
session_name: Name of the session
logs_parent_folder: Defines the parent folder where the togs will be stored.
```
## Others
```
start_tensorboard: If 1, it starts a tensorboard server 
tensorboard_port: Defines the tensorboard server port.
```



Agents
===========
The PRISMA framework provides 3 agents:

## DQN 

It implements the DQN (Deep Q-Network) agent. In this approach, a Neural Network is used in order to estimate the Q-value for a pair state-action. According to this approach, the input of the neural network (i.e., the observation space is composed by the packet's destination and the occupancy of the output buffers) 

## DQ-Routing

It implements the DQN agent for Q-Routing. In this approach, the observation space is composed by the packet's destination only. 


## SP

For using the Shortest Path algorithm, uncomment the block of lines in the launch_sp.sh file corresponding to this file.

Example
===========

We are going to illustrate Prisma usage with a pratical example. We will train a and develop a model for Q-Routing in two different topologies: Abilene and Geant. The Markov Decision Process (MDP) have the following formulation:

**Observation Space**: Packet's destination and interfaces buffer lenth (or tunnels delay - for overlay networks)

**Action Space**: Output's interface

**Reward**: Delay to the next hop

**Done**: If a packet arrives its final destination

In the above example, the agent's policy aims to minimize the reward (delay). The agent's model is described in the figure below. We expect the agents learn to efficiently route the packets and its performance might be proximal to a Shortest Path algorithm.

![My Image](images/q_routing_model.png)

The training curves are described below. The cost represents the average delay per packet which arrived to its final destination. The TD Error represents the model loss. We can observe that the model is capable of learning decreasing the loss over the training.

![My Image](images/training_curves.png)

In the figure below, we can evaluate the model performance. We can observe that the DQN-Routing agent is capable of learning a ploicy proximal to a Shortest Path algorithm. In some of the scenarios evaluated, it can performs better than the shortest path algorithm.

![My Image](images/testing_curves.png)
