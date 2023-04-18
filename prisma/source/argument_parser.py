#!/usr/bin python3
# -*- coding: utf-8 -*-


__author__ = "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__version__ = "0.1.0"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"


def parse_arguments():
    """ Retrieve and parse argument from the commandline
    """
    ### imports
    import argparse
    import networkx as nx
    import numpy as np
    import os
    import pathlib
    import json
    
    ## Setup the argparser
    description_txt = """PRISMA : Packet Routing Simulator for Multi-Agent Reinforcement Learning
                        PRISMA is a network simulation playground for developing and testing Multi-Agent Reinforcement Learning (MARL) solutions for dynamic packet routing (DPR). 
                        This framework is based on the OpenAI Gym toolkit and the Ns3 simulator.
                        """
    epilog_txt = """Examples:
                    For training a dqn routing model  :
                    python3 main.py --simTime=60 \
                                    --basePort=6555 \
                                    --train=1 \
                                    --agent_type="dqn_routing"\
                                    --session_name="train"\
                                    --logs_parent_folder=examples/geant/ \
                                    --traffic_matrix_path=examples/geant/traffic_matrices/node_intensity_normalized.txt \
                                    --adjacency_matrix_path=examples/geant/adjacency_matrix.txt \
                                    --node_coordinates_path=examples/geant/node_coordinates.txt \
                                    --training_step=0.007 \
                                    --batch_size=512 \
                                    --lr=1e-4 \
                                    --exploration_final_eps=0.1 \
                                    --exploration_initial_eps=1.0 \
                                    --iterationNum=3000 \
                                    --gamma=1 \
                                    --save_models=1 \
                                    --start_tensorboard=0 \
                                    --load_factor=0.5
                    """
    
    parser = argparse.ArgumentParser(prog='main.py', usage='python3 %(prog)s [options]', description=description_txt, epilog=epilog_txt, allow_abbrev=False)
    group1 = parser.add_argument_group('Global simulation arguments')
    group1.add_argument('--simTime', type=float, help='Simulation duration in seconds', default=60.0)
    group1.add_argument('--basePort', type=int, help='Starting port number', default=6555)
    group1.add_argument('--seed', type=int, help='Random seed used for the simulation', default=100)
    group1.add_argument('--train', type=int, help='If 1, train the model.Else, test it', default=1)
    group1.add_argument('--max_nb_arrived_pkts', type=int, help='If < 0, stops the episode at the provided number of arrived packets', default=-1)
    group1.add_argument('--ns3_sim_path', type=str, help='Path to the ns3-gym simulator folder', default="../ns3-gym/")
    group1.add_argument('--signalingSim', type=int, help='Allows the signaling in NS3 Simulation', default=0)
    group1.add_argument('--activateOverlay', type=int, help='Allows the signaling in NS3 Simulation in Overlay', default=1)
    group1.add_argument('--nPacketsOverlay', type=int, help='Allows the signaling in NS3 Simulation', default=2)
    group1.add_argument('--movingAverageObsSize', type=int, help="Sets the MA size of collecting the obs", default=5)
    group1.add_argument('--activateUnderlayTraffic', type=int, help="sets if there is underlay traffic", default=0)
    group1.add_argument('--activateUnderlayTrafficTrain', type=int, help="sets if there is underlay traffic", default=0)
    group1.add_argument('--map_overlay_path', type=str, help='Path to the map overlay file', default="mapOverlay_4n.txt")
    group1.add_argument('--pingAsObs', type=int, help="dets if ping value is used as observation", default=1)
    group1.add_argument('--groundTruthFrequence', type=float, help="groundTruthFrequence", default=0.1)
    group1.add_argument('--d_t_max_time', type=float, help="The maximum length in seconds of the digital twin database", default=5)


    group4 = parser.add_argument_group('Network parameters')
    group4.add_argument('--load_factor', type=float, help='scale of the traffic matrix', default=1)
    group4.add_argument('--topology_name', type=str, choices=["abilene", "geant"] ,help='Name of the network topology', default="abilene")
    group4.add_argument('--physical_adjacency_matrix_path', type=str, help='Path to the adjacency matrix', default="examples/abilene/adjacency_matrix.txt")
    group4.add_argument('--overlay_adjacency_matrix_path', type=str, help='Path to the adjacency matrix', default="examples/abilene/adjacency_matrix_2_5n.txt")
    group4.add_argument('--traffic_matrix_root_path', type=str, help='Path to the traffic matrix folder', default="examples/abilene/traffic_matrices/")
    group4.add_argument('--traffic_matrix_index', type=int, help='Index of the traffic matrix', default=0)
    group4.add_argument('--node_coordinates_path', type=str, help='Path to the nodes coordinates', default="examples/abilene/node_coordinates.txt")
    group4.add_argument('--max_out_buffer_size', type=int, help='Max nodes output buffer limit', default=30)
    group4.add_argument('--link_delay', type=str, help='Network links delay', default="0ms")
    group4.add_argument('--packet_size', type=int, help='Size of the packets in bytes', default=512)
    group4.add_argument('--link_cap', type=int, help='Network links capacity in bits per seconds', default=500000)
    


    group2 = parser.add_argument_group('Storing session logs arguments')
    group2.add_argument('--session_name', type=str, help='Name of the folder where to save the logs of the session', default=None)
    group2.add_argument('--logs_parent_folder', type=str, help='Name of the root folder where to save the logs of the sessions', default="examples/abilene/")
    group2.add_argument('--logging_timestep', type=int, help='Time delay (in real time) between each logging in seconds', default=5)
    group2.add_argument('--profile_session', type=int, help='If 1, the session is profiled', default=0)
    
    group3 = parser.add_argument_group('DRL Agent arguments')
    group3.add_argument('--agent_type', choices=["dqn_buffer", "dqn_routing", "dqn_buffer_fp", "dqn_buffer_lite", "dqn_buffer_lighter", "dqn_buffer_lighter_2", "dqn_buffer_lighter_3", "dqn_buffer_ff", "dqn_buffer_with_throughputs", "sp", "opt"], type=str, help='The type of the agent. Can be dqn_buffer, dqn_routing, dqn_buffer_fp, sp or opt', default="dqn_buffer")
    group3.add_argument('--signaling_type', type=str, choices=["NN", "target", "digital_twin", "ideal"], help='Type of the signaling. Can be "NN" for sending neighbors NN and (r,s\') tuple, "target" for sending only the target value and "ideal" for no signalisation (used when training)', default="ideal")
    group3.add_argument('--lr', type=float, help='Learning rate (used when training)', default=1e-4)
    group3.add_argument('--bigSignalingSize', type=int, help='Size of the neural network in bytes (used when signaling type is NN)', default=200)
    group3.add_argument('--prioritizedReplayBuffer', type=int, help='if true, use prioritized replay buffer using the gradient step as weights (used when training)', default=0)
    group3.add_argument('--smart_exploration', type=int, help='if true, explore using probability proportional to the inverse of the number of time the action was taken (used when training and exploration enabled)', default=0)
    group3.add_argument('--batch_size', type=int, help='Size of a batch (used when training)', default=512)
    group3.add_argument('--gamma', type=float, help='Gamma ratio for RL (used when training)', default=1)
    group3.add_argument('--iterationNum', type=int, help='Max iteration number for exploration (used when training)', default=3000)
    group3.add_argument('--exploration_initial_eps', type=float, help='Exploration intial value (used when training)', default=1.0)
    group3.add_argument('--exploration_final_eps', type=float, help='Exploration final value (used when training)', default=0.1)
    group3.add_argument('--load_path', type=str, help='Path to DQN models, if not None, loads the models from the given files', default=None)
    group3.add_argument('--save_models', type=int, help='if True, store the models at the end of the training', default=0)
    group3.add_argument('--snapshot_interval', type=int, help='Number of seconds between each snapshot of the models. If 0, desactivate snapshot', default=0)
    group3.add_argument('--training_step', type=float, help='Number of steps or seconds to train (used when training)', default=0.05)
    group3.add_argument('--sync_step', type=float, help='Number of seconds to sync NN if signaling_type is "NN". if -1, then compute it to have control/data of 10% (used when training)', default=1.0)
    group3.add_argument('--sync_ratio', type=float, help=' control/data ratio for computing the sync step automatically (used when training and sync step <0)', default=0.1)
    group3.add_argument('--replay_buffer_max_size', type=int, help='Max size of the replay buffers (used when training)', default=50000)
    group3.add_argument('--loss_penalty_type', type=str, choices=["None", "fixed", "constrained"],
                        help='Define the type of loss penalty to be added to the reward. If None, no loss penalty. If fixed, use a fixed loss pen. If constrained, use a loss mechanism based on RCPO',
                        default="fixed")
    group5 = parser.add_argument_group('Other parameters')
    group5.add_argument('--start_tensorboard', type=int, help='if True, starts a tensorboard server to keep track of simulation progress', default=0)
    group5.add_argument('--tensorboard_port', type=int, help='Tensorboard server port', default=16666)
    # parser.print_help()

    ## get the params dict 
    params = vars(parser.parse_args())

    ## add general env params depricated
    params["stepTime"]=0.1
    params["startSim"]=0
    params["simArgs"]={"--simTime": params["simTime"],
            "--testArg": 123}
    params["debug"]=0
    ## replace relative paths by absolute ones
    params["logs_parent_folder"] = os.path.abspath(params["logs_parent_folder"])
    if params["load_path"]:
        params["load_path"] = os.path.abspath(params["load_path"])
    params["physical_adjacency_matrix_path"] = os.path.abspath(params["physical_adjacency_matrix_path"])
    params["overlay_adjacency_matrix_path"] = os.path.abspath(params["overlay_adjacency_matrix_path"])
    params["map_overlay_path"] = os.path.abspath(params["map_overlay_path"])
    params["traffic_matrix_path"] = os.path.abspath(f'{params["traffic_matrix_root_path"].rstrip("/")}/node_intensity_normalized_{params["traffic_matrix_index"]}.txt')
    # params["traffic_matrix_path"] = os.path.abspath(f'{params["traffic_matrix_root_path"].rstrip("/")}/traffic_mat_{params["traffic_matrix_index"]}_adjusted_bps.txt')
    params["node_coordinates_path"] = os.path.abspath(params["node_coordinates_path"])
    params["ns3_sim_path"] = os.path.abspath(params["ns3_sim_path"])

    ## add the network topology to the params
    G=nx.DiGraph(nx.empty_graph())
    for i, element in enumerate(np.loadtxt(open(params["node_coordinates_path"]))):
        G.add_node(i,pos=tuple(element))
    G = nx.from_numpy_matrix(np.loadtxt(open(params["overlay_adjacency_matrix_path"])), parallel_edges=False, create_using=G)
    params["numNodes"] = G.number_of_nodes()
    print(G.number_of_nodes())
    params["G"] = G
    params["logs_parent_folder"] = params["logs_parent_folder"].rstrip("/")
    pathlib.Path(params["logs_parent_folder"]).mkdir(parents=True, exist_ok=True)

    ## Add session name
    if params["session_name"] == None:
        params["session_name"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    params["logs_folder"] = params["logs_parent_folder"] + "/" + params["session_name"]

    ## add some params for tensorboard writer
    params["global_stats_path"] = f'{params["logs_folder"]}/stats'
    params["nb_arrived_pkts_path"] = f'{params["logs_folder"]}/nb_arrived_pkts'
    params["nb_new_pkts_path"] = f'{params["logs_folder"]}/nb_new_pkts'
    params["nb_lost_pkts_path"] = f'{params["logs_folder"]}/nb_lost_pkts'

    ## Add optimal solution path
    topology_name = params["physical_adjacency_matrix_path"].split("/")[-3]
    # params["optimal_soltion_path"] = f"examples/{topology_name}/optimal_solution/11Nodes/{params['traffic_matrix_index']}_norm_matrix_uniform/{int(params['load_factor']*100)}_ut_minCostMCF.json"
    params["optimal_soltion_path"] = f"examples/{topology_name}/optimal_solution/{params['traffic_matrix_index']}_norm_matrix_uniform/{int(params['load_factor']*100)}_ut_minCostMCF.json"
    params["opt_rejected_path"] = os.path.abspath(f"examples/{topology_name}/optimal_solution/rejected_flows.txt")
    print(params["optimal_soltion_path"])
    if os.path.exists(params["optimal_soltion_path"]):
        np.savetxt(params["opt_rejected_path"], json.load(open(params["optimal_soltion_path"], "r"))["rejected_flows"], fmt='%.6f')
    else:
        print(f"WARNING: optimal solution file {params['optimal_soltion_path']} does not exist")
        np.savetxt(params["opt_rejected_path"], np.zeros((params["numNodes"], params["numNodes"])), fmt='%.6f')
    # params["optimal_soltion_path"] = f"examples/{topology_name}/optimal_solution/{params['traffic_matrix_index']}_adjusted_5_nodes_mesh_norm_matrix_uniform/{int(params['load_factor']*100)}_ut_minCostMCF.json"
    
    ## compute the loss penalty
    # params["loss_penalty"] = ((((params["max_out_buffer_size"] + 1)*params["packet_size"]*8)/params["link_cap"])) *params["numNodes"]
    params["loss_penalty"] = ((((params["max_out_buffer_size"] + 512+30)*8)/params["link_cap"])+0.001)* params["numNodes"]
    
    return params