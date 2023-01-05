#!/usr/bin python3
# -*- coding: utf-8 -*-


__author__ = "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__version__ = "0.1.0"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"

### imports
import argparse
from time import sleep, time
import numpy as np
import threading
import copy
import tensorflow as tf
import random
import networkx as nx
import os
import datetime
import csv
import json
from tensorboard.plugins.hparams import api as hp
from source.agent import Agent
from source.utils import save_model
import subprocess, signal
import shlex
from tensorboard.plugins.custom_scalar import summary as cs_summary
from tensorboard.plugins.custom_scalar import layout_pb2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# tf.config.set_soft_device_placement(True)

def arguments_parser():
    """ Retrieve and parse argument from the commandline
    """
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
                                    --training_trigger_type="time" \
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

    group4 = parser.add_argument_group('Network parameters')
    group4.add_argument('--load_factor', type=float, help='scale of the traffic matrix', default=1)
    group4.add_argument('--load_factor_trainning', type=float, help='scale of the traffic matrix', default=1)
    group4.add_argument('--adjacency_matrix_path', type=str, help='Path to the adjacency matrix', default="examples/abilene/adjacency_matrix.txt")
    group4.add_argument('--overlay_matrix_path', type=str, help='Path to the adjacency matrix', default="examples/abilene/overlay_matrix_4n.txt")
    group4.add_argument('--agent_adjacency_matrix_path', type=str, help='Path to the adjacency matrix', default="examples/abilene/adjacency_matrix_2_4n.txt")
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
    
    group3 = parser.add_argument_group('DRL Agent arguments')
    group3.add_argument('--agent_type', choices=["dqn_buffer", "dqn_routing", "dqn_buffer_fp", "dqn_buffer_lite", "sp", "opt"], type=str, help='The type of the agent. Can be dqn_buffer, dqn_routing, dqn_buffer_fp, sp or opt', default="dqn_buffer")
    group3.add_argument('--signaling_type', type=str, choices=["NN", "target", "ideal"], help='Type of the signaling. Can be "NN" for sending neighbors NN and (r,s\') tuple, "target" for sending only the target value and "ideal" for no signalisation (used when training)', default="ideal")
    group3.add_argument('--lr', type=float, help='Learning rate (used when training)', default=1e-4)
    group3.add_argument('--prioritizedReplayBuffer', type=int, help='if true, use prioritized replay buffer using the gradient step as weights (used when training)', default=0)
    
    group3.add_argument('--batch_size', type=int, help='Size of a batch (used when training)', default=512)
    group3.add_argument('--gamma', type=float, help='Gamma ratio for RL (used when training)', default=1)
    group3.add_argument('--iterationNum', type=int, help='Max iteration number for exploration (used when training)', default=3000)
    group3.add_argument('--exploration_initial_eps', type=float, help='Exploration intial value (used when training)', default=1.0)
    group3.add_argument('--exploration_final_eps', type=float, help='Exploration final value (used when training)', default=0.1)
    group3.add_argument('--load_path', type=str, help='Path to DQN models, if not None, loads the models from the given files', default=None)
    group3.add_argument('--save_models', type=int, help='if True, store the models at the end of the training', default=0)
    group3.add_argument('--training_trigger_type', type=str, choices=["event", "time"], help='Type of the training trigger, can be "event" (for event based) or "time" (for time based) (used when training)', default="time")
    group3.add_argument('--training_step', type=float, help='Number of steps or seconds to train (used when training)', default=0.05)
    group3.add_argument('--sync_step', type=float, help='Number of seconds to sync NN if signaling_type is "NN". if -1, then compute it to have control/data of 10% (used when training)', default=1.0)
    group3.add_argument('--sync_ratio', type=float, help=' control/data ratio for computing the sync step automatically (used when training and sync step <0)', default=0.1)
    group3.add_argument('--replay_buffer_max_size', type=int, help='Max size of the replay buffers (used when training)', default=50000)

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
    #if params["save_models"]:
    #    params["save_models"] = os.path.abspath(params["save_models"])
    params["adjacency_matrix_path"] = os.path.abspath(params["adjacency_matrix_path"])
    params["agent_adjacency_matrix_path"] = os.path.abspath(params["agent_adjacency_matrix_path"])
    params["opt_rejected_path"] = os.path.abspath("test.txt")
    params["map_overlay_path"] = os.path.abspath(params["map_overlay_path"])
    params["overlay_matrix_path"] = os.path.abspath(params["overlay_matrix_path"])
    params["traffic_matrix_path"] = os.path.abspath(f'{params["traffic_matrix_root_path"].rstrip("/")}/node_intensity_normalized_{params["traffic_matrix_index"]}.txt')
    #params["traffic_matrix_path"] = os.path.abspath(f'{params["traffic_matrix_root_path"].rstrip("/")}/traffic_mat_{params["traffic_matrix_index"]}_adjusted_bps.txt')
    params["node_coordinates_path"] = os.path.abspath(params["node_coordinates_path"])
    params["ns3_sim_path"] = os.path.abspath(params["ns3_sim_path"])

    ## add the network topology to the params
    G=nx.DiGraph(nx.empty_graph())
    for i, element in enumerate(np.loadtxt(open(params["node_coordinates_path"]))):
        G.add_node(i,pos=tuple(element))
    G = nx.from_numpy_matrix(np.loadtxt(open(params["agent_adjacency_matrix_path"])), parallel_edges=False, create_using=G)
    params["maxNumNodes"] = 11
    params["numNodes"] = G.number_of_nodes()
    print(G.number_of_nodes())
    params["G"] = G
    params["logs_parent_folder"] = params["logs_parent_folder"].rstrip("/")

    ## Add session name
    if params["session_name"] == None:
        params["session_name"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    params["logs_folder"] = params["logs_parent_folder"] + "/results/"+ params["session_name"]

    ## add some params for tensorboard writer
    params["global_stats_path"] = f'{params["logs_folder"]}/stats'
    params["nb_arrived_pkts_path"] = f'{params["logs_folder"]}/nb_arrived_pkts'
    params["nb_new_pkts_path"] = f'{params["logs_folder"]}/nb_new_pkts'
    params["nb_lost_pkts_path"] = f'{params["logs_folder"]}/nb_lost_pkts'
    
    ## Add optimal solution path
    topology_name = params["adjacency_matrix_path"].split("/")[-2]
    params["optimal_soltion_path"] = f"examples/{topology_name}/optimal_solution/{params['traffic_matrix_index']}_adjusted_5_nodes_mesh_norm_matrix_uniform/{int(params['load_factor']*100)}_ut_minCostMCF.json"
    return params

def custom_plots():
    """define the costume plots for tensorboard"
    """
    cs = cs_summary.pb(
            layout_pb2.Layout(
                category=[
                    layout_pb2.Category(
                        title="Main evaluation metrics",
                        chart=[
                            layout_pb2.Chart(
                                title="Avg Delay per arrived pkts",
                                multiline=layout_pb2.MultilineChartContent(tag=[r"avg_delay_over_time"])),
                            layout_pb2.Chart(
                                title="Avg Cost per arrived pkts",
                                multiline=layout_pb2.MultilineChartContent(tag=[r"avg_cost_over_time"])),
                            layout_pb2.Chart(
                                title="Loss Ratio",
                                multiline=layout_pb2.MultilineChartContent(tag=[r"loss_ratio_over_time"])),
                        ]),
                    # layout_pb2.Category(
                    #     title="Global info about the env",
                    #     chart=[
                    #         layout_pb2.Chart(
                    #             title="Average hops over time",
                    #             multiline=layout_pb2.MultilineChartContent(tag=[r"avg_hops_over_time"])),
                    #         layout_pb2.Chart(
                    #             title="total rewards with and without loss",
                    #             multiline=layout_pb2.MultilineChartContent(tag=[r"(total_rewards_with_loss_over_time|total_rewards_over_time)"])),
                    #         layout_pb2.Chart(
                    #             title="Buffers occupation",
                    #             multiline=layout_pb2.MultilineChartContent(tag=[r"nb_buffered_pkts_over_time"])),
                    #         layout_pb2.Chart(
                    #             title="new pkts vs lost pkts vs arrived pkts",
                    #             multiline=layout_pb2.MultilineChartContent(tag=[r"(total_new_rcv_pkts_over_time | total_lost_pkts_over_time | total_arrived_pkts_over_time)"])),
                    #     ]),
                    layout_pb2.Category(
                        title="Training metrics",
                        chart=[
                            layout_pb2.Chart(
                                title="Td error",
                                multiline=layout_pb2.MultilineChartContent(tag=[r"MSE_loss_over_time"])),
                            layout_pb2.Chart(
                                title="exploration value",
                                multiline=layout_pb2.MultilineChartContent(tag=[r"exploaration_value_over_time"])),
                            layout_pb2.Chart(
                                title="replay buffers length",
                                multiline=layout_pb2.MultilineChartContent(tag=[r"replay_buffer_length_over_time"])),
                        ]),
                ]
            )
        )
    return cs

def stats_writer(summary_writer_session, summary_writer_nb_arrived_pkts, summary_writer_nb_lost_pkts, summary_writer_nb_new_pkts):
    """ Write the stats of the session to the logs dir using tensorboard writer
    Args:
        summary_writer_session: main session writer for the reward, loss, delay and nb buffered pkts
        summary_writer_nb_arrived_pkts: writer for nb arrived pkts
        summary_writer_nb_lost_pkts: writer for nb lost pkts
        summary_writer_nb_new_pkts: writer for nb new pkts
    """
    ## write the global stats
    if Agent.sim_injected_packets > 0:
        loss_ratio = Agent.sim_dropped_packets/Agent.sim_injected_packets
    else:
        loss_ratio = -1
    if Agent.sim_delivered_packets > 0:
        avg_delay = Agent.sim_avg_e2e_delay/(Agent.sim_delivered_packets*1000)
        avg_cost = Agent.sim_cost/(Agent.sim_delivered_packets+Agent.sim_dropped_packets)
        avg_hops = Agent.total_hops/Agent.sim_delivered_packets
    else:
        avg_delay = -1
        avg_cost = -1
        avg_hops = -1

    with summary_writer_session.as_default():
        ## total rewards
        tf.summary.scalar('total_e2e_delay_over_iterations', Agent.total_e2e_delay, step=Agent.currIt)
        tf.summary.scalar('total_e2e_delay_over_time', Agent.total_e2e_delay, step=int(Agent.curr_time*1e6))
        tf.summary.scalar('total_rewards_with_loss_over_iterations', Agent.total_rewards_with_loss, step=Agent.currIt)
        tf.summary.scalar('total_rewards_with_loss_over_time', Agent.total_rewards_with_loss, step=int(Agent.curr_time*1e6))
        ## loss ratio
        tf.summary.scalar('loss_ratio_over_time', loss_ratio, step=int(Agent.curr_time*1e6))
        tf.summary.scalar('loss_ratio_over_iterations', loss_ratio, step=Agent.currIt)
        ## total hops and avg hops
        tf.summary.scalar('total_hops_over_iterations', Agent.total_hops, step=Agent.currIt)
        tf.summary.scalar('total_hops_over_time', Agent.total_hops, step=int(Agent.curr_time*1e6))
        tf.summary.scalar('avg_hops_over_iterations', avg_hops, step=Agent.currIt)
        tf.summary.scalar('avg_hops_over_time', avg_hops, step=int(Agent.curr_time*1e6))
        tf.summary.scalar('ma_avg_hops_over_iterations', np.array(Agent.nb_hops).mean(), step=Agent.currIt)
        tf.summary.scalar('ma_avg_hops_over_time', np.array(Agent.nb_hops).mean(), step=int(Agent.curr_time*1e6))
        ## buffers occupation
        tf.summary.scalar('nb_buffered_pkts_over_time', Agent.sim_buffered_packets, step=int(Agent.curr_time*1e6))
        tf.summary.scalar('nb_buffered_pkts_over_iterations', Agent.sim_buffered_packets, step=Agent.currIt)
        ## avg cost and avg delay
        tf.summary.scalar('avg_cost_over_iterations', avg_cost, step=Agent.currIt)
        tf.summary.scalar('avg_cost_over_time', avg_cost, step=int(Agent.curr_time*1e6))
        tf.summary.scalar('avg_delay_over_iterations', avg_delay, step=Agent.currIt)
        tf.summary.scalar('avg_delay_over_time', avg_delay, step=int(Agent.curr_time*1e6))
        tf.summary.scalar('ma_delays_over_iterations', np.array(Agent.delays).mean(), step=Agent.currIt)
        tf.summary.scalar('ma_delays_over_time', np.array(Agent.delays).mean(), step=int(Agent.curr_time*1e6))

    with summary_writer_nb_arrived_pkts.as_default():
        tf.summary.scalar('pkts_over_iterations', Agent.sim_delivered_packets, step=Agent.currIt)
        tf.summary.scalar('pkts_over_time', Agent.sim_delivered_packets, step=int(Agent.curr_time*1e6))

    with summary_writer_nb_lost_pkts.as_default():
        tf.summary.scalar('pkts_over_iterations', Agent.sim_dropped_packets, step=Agent.currIt)
        tf.summary.scalar('pkts_over_time', Agent.sim_dropped_packets, step=int(Agent.curr_time*1e6))

    with summary_writer_nb_new_pkts.as_default():
        tf.summary.scalar('pkts_over_iterations', Agent.sim_injected_packets, step=Agent.currIt)
        tf.summary.scalar('pkts_over_time', Agent.sim_injected_packets, step=int(Agent.curr_time*1e6))

def run_ns3(params):
    """
    Run the ns3 simulator
    Args: 
        params(dict): parameter dict
    """ 
    ## check if ns3-gym is in the folder
    if "waf" not in os.listdir(params["ns3_sim_path"]):
        raise Exception(f'Unable to locate ns3-gym in the folder : {params["ns3_sim_path"]}')
        
    ## store current folder path
    current_folder_path = os.getcwd()

    ## Copy prisma into ns-3 folder
    os.system(f'rsync -r ./ns3/* {params["ns3_sim_path"].rstrip("/")}/scratch/prisma')
    os.system(f'rsync -r ./ns3_model/ipv4-interface.cc {params["ns3_sim_path"].rstrip("/")}/src/internet/model')

    ## go to ns3 dir
    os.chdir(params["ns3_sim_path"])
    
    ## run ns3 configure
    #configure_command = './waf -d optimized configure'
    os.system('./waf configure')
    print(params['agent_type'])
    ## run NS3 simulator
    ns3_params_format = ('prisma --simSeed={} --openGymPort={} --simTime={} --AvgPacketSize={} '
                        '--LinkDelay={} --LinkRate={} --MaxBufferLength={} --load_factor={} '
                        '--adj_mat_file_name={} --overlay_mat_file_name={} --node_coordinates_file_name={} '
                        '--node_intensity_file_name={} --signaling={} --AgentType={} --signalingType={} '
                        '--syncStep={} --lossPenalty={} --activateOverlaySignaling={} --nPacketsOverlaySignaling={} '
                        '--train={} --movingAverageObsSize={} --activateUnderlayTraffic={} --opt_rejected_file_name={} --map_overlay_file_name={}'.format( params["seed"],
                                             params["basePort"],
                                             str(params["simTime"]),
                                             params["packet_size"],
                                             params["link_delay"],
                                             str(params["link_cap"]) + "bps",
                                             str(params["max_out_buffer_size"]) + "B",
                                             params["load_factor"],
                                             params["adjacency_matrix_path"],
                                             params["overlay_matrix_path"],
                                             params["node_coordinates_path"],
                                             params["traffic_matrix_path"],
                                             bool(params["signalingSim"]),
                                             params["agent_type"],
                                             params["signaling_type"],
                                             params["sync_step"],
                                             params["loss_penalty"],
                                             bool(params["activateOverlay"]),
                                             params["nPacketsOverlay"],
                                             bool(params["train"]),
                                             params["movingAverageObsSize"],
                                             bool(params["activateUnderlayTraffic"]),
                                             params["opt_rejected_path"],
                                             params["map_overlay_path"]
                                             ))
    run_ns3_command = shlex.split(f'./waf --run "{ns3_params_format}"')
    subprocess.Popen(run_ns3_command)
    os.chdir(current_folder_path)

def main():
    ## Get the arguments from the parser
    params = arguments_parser()

    ## Metrics
    # params["METRICS"] = ["avg_delay", "loss_ratio", "reward"]

    ## compute the loss penalty
    # params["loss_penalty"] = ((((params["max_out_buffer_size"] + 1)*params["packet_size"]*8)/params["link_cap"])) *params["numNodes"]
    params["loss_penalty"] = ((((params["max_out_buffer_size"] + 512+30)*8)/params["link_cap"])+0.001)*params["maxNumNodes"]

    ## fix the seed
    tf.random.set_seed(params["seed"])
    np.random.seed(params["seed"])
    random.seed(params["seed"])
    
    ## test results file name
    test_results_file_name = f'{params["logs_parent_folder"]}/_tests_overlay_5n_2/ter_t_1000_20k_tr_{params["traffic_matrix_index"]}_underlayTraff_{params["activateUnderlayTrafficTrain"]}_{params["agent_type"]}_{params["signaling_type"]}_{params["signalingSim"]}_fixed_rb_{params["replay_buffer_max_size"]}_sync{int(1000*params["sync_step"])}ms_ratio_{int(100*params["sync_ratio"])}_overlayPackets_{params["nPacketsOverlay"]}_loadTrain_{int(100*params["load_factor_trainning"])}_load_{int(100*params["load_factor"])}.txt'
    train_results_file_name = f'{params["logs_parent_folder"]}/_train_overlay_5n/ter_1000_20k_tr_{params["traffic_matrix_index"]}_underlayTraff_{params["activateUnderlayTrafficTrain"]}_{params["agent_type"]}_{params["signaling_type"]}_{params["signalingSim"]}_fixed_rb_{params["replay_buffer_max_size"]}_sync{int(1000*params["sync_step"])}ms_ratio_{int(100*params["sync_ratio"])}_overlayPackets_{params["nPacketsOverlay"]}_loadTrain_{int(100*params["load_factor_trainning"])}_load_{int(100*params["load_factor"])}.txt'
    print(test_results_file_name)
    if params["train"] == 1:
        if params["session_name"] in os.listdir(params["logs_parent_folder"] + "/saved_models/"):
                print(f'The couple {params["seed"]} {params["traffic_matrix_index"]} already exists in : {params["logs_parent_folder"] + "/saved_models/" + params["session_name"]}')
                return 1
         
    else:
        if test_results_file_name.split("/")[-1] in os.listdir("/".join(test_results_file_name.split("/")[:-1])):
            ## check if the couple seed traff mat idx is already in the file
            if [params["traffic_matrix_index"], params["seed"]] in np.atleast_2d(np.genfromtxt(test_results_file_name, delimiter=",", dtype=int))[:, 2:4].tolist():
                print(f'The couple {params["seed"]} {params["traffic_matrix_index"]} already exists in the {test_results_file_name}')
                return 1
                        
    # if params["train"] == 0:
    #     params["signalingSim"] = 0
    ## Setup writer for the global stats
    summary_writer_parent = tf.summary.create_file_writer(logdir=params["logs_folder"] )
    summary_writer_session = tf.summary.create_file_writer(logdir=params["global_stats_path"] )
    summary_writer_nb_arrived_pkts = tf.summary.create_file_writer(logdir=params["nb_arrived_pkts_path"] )
    summary_writer_nb_new_pkts = tf.summary.create_file_writer(logdir=params["nb_new_pkts_path"] )
    summary_writer_nb_lost_pkts = tf.summary.create_file_writer(logdir=params["nb_lost_pkts_path"] )

    ## write the session info
    with tf.summary.create_file_writer(logdir=params["logs_folder"]).as_default():
        ## Adapt the dict to the hparams api
        dict_to_store = copy.deepcopy(params)
        dict_to_store["G"] = str(params["G"])
        dict_to_store["load_path"] = str(params["load_path"])
        dict_to_store["simArgs"] = str(params["simArgs"])
        hp.hparams(dict_to_store)  # record the values used in this trial
    
    ## Define the custom categories in tensorboard
    with summary_writer_parent.as_default():
        tf.summary.experimental.write_raw_pb(
                custom_plots().SerializeToString(), step=0
            )
    ## setup the agents (fix the static variables)
    Agent.init_static_vars(params)
    
    print("running ns-3")
    ## run ns3 simulator
    run_ns3(params)
    
    

    ## run the agents threads
    agent_instances = []
    print(params["G"].nodes())
    for index in params["G"].nodes(): #range(params["numNodes"]):
        print("Index", index)
        agent_instance = Agent(index, agent_type=params["agent_type"], train=params["train"])
        agent_instances.append(agent_instance)
        th1 = threading.Thread(target=agent_instance.run_forwarder, args=())
        th1.start()
        if params["train"]:
            th2 = threading.Thread(target=agent_instance.run_trainer, args=(params["training_trigger_type"],))
            th2.start()

    ## Run tensorboard server
    if params["start_tensorboard"]:
        args = shlex.split(f'python3 -m tensorboard.main --logdir={params["logs_folder"]} --port={params["tensorboard_port"]}')
        subprocess.Popen(args)
    sleep(1)
    
    print(params)
    ## wait until simulation complete and update info about the env at each timestep
    while threading.active_count() > params["numNodes"] * (1+ params["train"]):
        sleep(params["logging_timestep"])
        stats_writer(summary_writer_session, summary_writer_nb_arrived_pkts, summary_writer_nb_lost_pkts, summary_writer_nb_new_pkts)

    print(f"Signaling overhead = {Agent.small_signaling_overhead_counter}")
    print(f""" Summary of the Simulation:
            Total number of packets = {Agent.sim_injected_packets}, 
            Number of arrived packets = {Agent.sim_delivered_packets},
            Number of lost packets = {Agent.sim_dropped_packets},
            Number Test Drooped = {Agent.sim_test_dropped},
            Delay_real = {Agent.sim_avg_e2e_delay},
            Cost = {Agent.sim_cost},
            Hops = {Agent.total_hops/Agent.sim_delivered_packets},
            nbBytesData = {Agent.sim_bytes_data},
            nbBytesBigSignaling = {Agent.sim_bytes_big_signaling},
            nbBytesSmallSignaling = {Agent.sim_bytes_small_signaling},
            nbBytesOverlaySignalingForward = {Agent.sim_bytes_overlay_signaling_forward},
            nbBytesOverlaySignalingBack = {Agent.sim_bytes_overlay_signaling_back},
            OverheadRatio = {(Agent.sim_bytes_big_signaling+Agent.sim_bytes_small_signaling+Agent.sim_bytes_overlay_signaling_forward+Agent.sim_bytes_overlay_signaling_back)/Agent.sim_bytes_data}
            Number of Total packets = {Agent.sim_global_injected_packets},
            Number Total Drooped = {Agent.sim_global_dropped_packets}
            """)
    
    print(f""" Summary of the episode :
            Total number of Iterations = {Agent.currIt},
            Total number of Transitions = {Agent.nb_transitions},
            Simulation time = {Agent.curr_time},
            Total e2e delay = {Agent.total_e2e_delay}, 
            Total number of packets = {Agent.total_new_rcv_pkts}, 
            Number of arrived packets = {Agent.total_arrived_pkts},
            Number of lost packets = {Agent.node_lost_pkts},
            Loss ratio = {Agent.node_lost_pkts/Agent.total_new_rcv_pkts}
            Delay_ideal = {np.array(Agent.delays_ideal).mean()}
            Delay_real = {np.array(Agent.delays_real).mean()}
            total cost = {Agent.total_rewards_with_loss}
            theoretical cost = {((Agent.node_lost_pkts * Agent.loss_penalty) + np.array(Agent.delays_ideal).sum())/(Agent.node_lost_pkts + Agent.total_arrived_pkts)}
            Avg cost = {Agent.total_rewards_with_loss/Agent.total_new_rcv_pkts}
            Reward = {np.array(Agent.rewards).mean()} 
            Signaling overhead = {Agent.small_signaling_overhead_counter + Agent.big_signaling_overhead_counter}
            small nb Signaling pkts = {Agent.small_signaling_overhead_counter}
            big nb Signaling pkts ideal = {Agent.big_signaling_overhead_counter}
            Data pkts size = {Agent.total_data_size}
    
            """)
    if Agent.total_arrived_pkts:
        print(f"Average delay per arrived packets = {Agent.total_e2e_delay/(Agent.total_arrived_pkts*1000)}")
        
    fields_stats=[params["agent_type"],
                        params["signaling_type"], 
                        params["traffic_matrix_index"], 
                        params["seed"],
                        params["replay_buffer_max_size"],
                        params["sync_step"],
                        Agent.node_lost_pkts/Agent.total_new_rcv_pkts,
                        np.array(Agent.delays_ideal).mean(),
                        ((Agent.node_lost_pkts * Agent.loss_penalty) + np.array(Agent.delays_ideal).sum())/(Agent.node_lost_pkts + Agent.total_arrived_pkts),
                        Agent.total_rewards_with_loss/Agent.total_new_rcv_pkts,
                        Agent.total_rewards_with_loss,
                        Agent.small_signaling_overhead_counter + Agent.big_signaling_overhead_counter,
                        Agent.small_signaling_overhead_counter,
                        Agent.big_signaling_overhead_counter,
                        Agent.total_new_rcv_pkts,
                        Agent.total_arrived_pkts,
                        Agent.node_lost_pkts,
                        Agent.total_data_size,
                        Agent.nb_transitions
                    ]
    new_fields_stats=[params["agent_type"],
                        params["signaling_type"], 
                        params["traffic_matrix_index"], 
                        params["seed"],
                        params["replay_buffer_max_size"],
                        params["sync_step"],
                        params["load_factor"],
                        Agent.sim_dropped_packets/Agent.sim_injected_packets,
                        Agent.sim_injected_packets,
                        Agent.sim_delivered_packets,
                        Agent.sim_dropped_packets,
                        Agent.sim_avg_e2e_delay,
                        Agent.sim_cost,
                        Agent.total_hops/Agent.sim_delivered_packets,
                        (Agent.sim_bytes_big_signaling+Agent.sim_bytes_small_signaling+Agent.sim_bytes_overlay_signaling_forward+Agent.sim_bytes_overlay_signaling_back)/Agent.sim_bytes_data,
                        Agent.sim_global_injected_packets,
                        Agent.sim_global_dropped_packets,
                        Agent.sim_global_dropped_packets/Agent.sim_global_injected_packets
                        ]
                          
        
    if params["train"] == 0:
           
        with open(test_results_file_name, 'a') as f: 
            writer = csv.writer(f) 
            writer.writerow(new_fields_stats)
    else:   
        with open(train_results_file_name, 'a') as f: 
            writer = csv.writer(f) 
            writer.writerow(new_fields_stats) 
    

    ## save models        
    if params["save_models"] and Agent.curr_time >= params["simTime"]-5:
        save_model(Agent.agents, params["G"].nodes(), params["session_name"], 1, 1, root=params["logs_parent_folder"] + "/saved_models/")
        if params["agent_type"] == "dqn_buffer_fp":
            for item in agent_instances:
                np.savetxt(f'{params["logs_parent_folder"].rstrip("/")}/saved_models/{params["session_name"]}/node_{item.index}_final_params.txt',  [item.update_eps.numpy().item(), item.gradient_step_idx])
    
    
    ## saving the replay buffers
    # for i, rb in enumerate(Agent.replay_buffer):
    #     rb.save(f"rb_savings/{i}.txt")
    ## saving the transition array
    #for node_idx in range(Agent.numNodes):
    #    if(not os.path.exists(f"lock_files/{params['session_name']}")):  
    #        os.mkdir(f"lock_files/{params['session_name']}/") 
    #    np.savetxt(f"lock_files/{params['session_name']}/{node_idx}.txt", np.array(Agent.lock_info_array[node_idx], dtype=object), fmt = "%s", header = "src dst node next_hop ideal_time real_time obs action")


if __name__ == '__main__':
    ## create a process group
    import traceback
    os.setpgrp()
    try:
        print("starting process group")
        start_time = time()
        main()
        print("Elapsed time = ", str(datetime.timedelta(seconds= time() - start_time)))
    except:
        traceback.print_exc()
    finally:
        os.killpg(0, signal.SIGKILL)