#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from tensorboard.plugins.hparams import api as hp
import multiprocessing
from source.agent_action import Agent
import subprocess, signal
import shlex
from tensorboard.plugins.custom_scalar import summary as cs_summary
from tensorboard.plugins.custom_scalar import layout_pb2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def arguments_parser():
    """ Retrieve and parse argument from the commandline
    """
    ## Setup the argparser
    parser = argparse.ArgumentParser(prog='Multi Agents Threaded NS3', usage='python3 %(prog)s [options]')
    parser.add_argument('--simTime', type=float, help='Simulation duration in seconds', default=60.0)
    parser.add_argument('--session_name', type=str, help='Name of the folder where to save the logs of the session', default=None)
    parser.add_argument('--logging_timestep', type=int, help='Time delay (in real time) between each logging in seconds', default=15)
    parser.add_argument('--basePort', type=int, help='Starting port number', default=6555)
    parser.add_argument('--agent_type', choices=["dqn", "sp", "opt"], type=str, help='The type of the agent. Can be dqn, sp or opt', default="dqn")
    parser.add_argument('--train', type=int, help='Whether to train the model or test it', default=1)
    parser.add_argument('--seed', type=int, help='Random seed used for the simulation', default=1)
    parser.add_argument('--max_out_buffer_size', type=int, help='Max nodes output buffer limit', default=30)
    parser.add_argument('--packet_size', type=int, help='Size of the packets in bytes', default=512)
    parser.add_argument('--link_delay', type=str, help='Network links delay', default="2ms")
    parser.add_argument('--lr', type=float, help='Learning rate (used when training)', default=1e-3)
    parser.add_argument('--batch_size', type=int, help='Size of a batch (used when training)', default=128)
    parser.add_argument('--gamma', type=float, help='Gamma ratio for RL (used when training)', default=1)
    parser.add_argument('--iterationNum', type=int, help='Max iteration number for exploration (used when training)', default=1000)
    parser.add_argument('--exploration_initial_eps', type=float, help='Exploration intial value (used when training)', default=0.5)
    parser.add_argument('--exploration_final_eps', type=float, help='Exploration final value (used when training)', default=0.1)
    parser.add_argument('--load_path', type=str, help='Path to DQN models, if not None, loads the models from the given files', default=None)
    parser.add_argument('--save_path', type=str, help='Name of the folder where to store the models at the end of the training', default=None)
    parser.add_argument('--link_cap', type=int, help='Network links capacity in bits per seconds', default=500000)
    parser.add_argument('--training_freq', type=int, help='Number of timesteps to train (used when training)', default=16)
    parser.add_argument('--replay_buffer_max_size', type=int, help='Max size of the replay buffers (used when training)', default=10000)
    parser.add_argument('--load_factor', type=float, help='scale of the traffic matrix', default=0.01)
    parser.add_argument('--adjacency_matrix_path', type=str, help='Path to the adjacency matrix', default="scratch/my_network/adjacency_matrix.txt")
    parser.add_argument('--node_coordinates_path', type=str, help='Path to the nodes coordinates', default="scratch/my_network/node_coordinates.txt")
    parser.add_argument('--traffic_matrix_path', type=str, help='Path to the traffic matrix file', default="scratch/my_network/node_intensity.txt")
    # parser.print_help()


    return vars(parser.parse_args())

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
                                multiline=layout_pb2.MultilineChartContent(tag=[r"avg_delay_over_time"])),
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

def write_stats():
    """ Write the stats of the session to the logs dir using tensorboard writer
    """
    pass


def main():
    ## Get the arguments from the parser
    params = arguments_parser()

    ## Add constant params
    params["logs_parent_folder"] = "outputs/"
    ## Metrics
    # params["METRICS"] = ["avg_delay", "loss_ratio", "reward"]
    ## general env params depricated
    params["stepTime"]=0.1
    params["startSim"]=0
    params["simArgs"]={"--simTime": params["simTime"],
            "--testArg": 123}
    params["debug"]=0
    ## compute the loss penalty
    params["loss_penalty"] = ((params["max_out_buffer_size"] + 1)*params["packet_size"]*8)/params["link_cap"]
    ## network topology
    G=nx.Graph()
    os.chdir("../../")
    for i, element in enumerate(np.loadtxt(open(params["node_coordinates_path"]))):
        G.add_node(i,pos=tuple(element))
    G = nx.from_numpy_matrix(np.loadtxt(open(params["adjacency_matrix_path"])), create_using=G)
    params["numNodes"] = G.number_of_nodes()
    params["G"] = G
    os.chdir("scratch/my_network")
    if params["session_name"] == None:
        params["session_name"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    params["logs_folder"] = params["logs_parent_folder"] + params["session_name"]

    # between_train_steps_delay = 0.05

    ## fix the seed
    tf.random.set_seed(params["seed"])
    np.random.seed(params["seed"])
    random.seed(params["seed"])


    ## setup writer for the global stats 
    global_stats_path = f'{params["logs_folder"]}/stats'
    params["global_stats_path"] = global_stats_path

    summary_writer_parent = tf.summary.create_file_writer(logdir=params["logs_parent_folder"])
    summary_writer_session = tf.summary.create_file_writer(logdir=global_stats_path)


    ## write the session info
    with tf.summary.create_file_writer(logdir=params["logs_folder"]).as_default():
        ## adapt the dict to the hparams api
        dict_to_store = copy.deepcopy(params)
        dict_to_store["G"] = str(G)
        dict_to_store["load_path"] = str(params["load_path"])
        dict_to_store["save_path"] = str(params["save_path"])
        dict_to_store["simArgs"] = str(params["simArgs"])
        hp.hparams(dict_to_store)  # record the values used in this trial
    
    ## run NS3 simulator
    os.chdir("../../")
    ns3_params_format = ('my_network --simSeed={} --openGymPort={} --simTime={} --AvgPacketSize={} '
                        '--LinkDelay={} --LinkRate={} --MaxBufferLength={} --load_factor={} '
                        '--adj_mat_file_name={} --node_coordinates_file_name={} --node_intensity_file_name={}'.format( params["seed"],
                                                                                                                        params["basePort"],
                                                                                                                        str(params["simTime"]),
                                                                                                                        params["packet_size"],
                                                                                                                        params["link_delay"],
                                                                                                                        str(params["link_cap"]) + "bps",
                                                                                                                        str(params["max_out_buffer_size"]) + "p",
                                                                                                                        params["load_factor"],
                                                                                                                        params["adjacency_matrix_path"],
                                                                                                                        params["node_coordinates_path"],
                                                                                                                        params["traffic_matrix_path"]))
    args = shlex.split(f'./waf --run "{ns3_params_format}"')
    subprocess.Popen(args)
    os.chdir("scratch/my_network")

    ## setup the agents (fix the static variables)
    Agent.init_static_vars(params)

    ## run the agents
    for index in range(params["numNodes"]):
        agent_instance = Agent(index, agent_type=params["agent_type"], train=params["train"])
        x = threading.Thread(target=agent_instance.run, args=())
        x.start()

    ## Define the custom categories in tensorboard
    with summary_writer_parent.as_default():
        tf.summary.experimental.write_raw_pb(
                custom_plots().SerializeToString(), step=0
            )

    ## Run tensorboard server
    args = shlex.split(f'tensorboard --logdir={params["logs_parent_folder"]} --port=16666')
    subprocess.Popen(args)
    sleep(1)

    ## wait until simulation complete and update info about the env at each timestep
    while threading.active_count() > 1:

        sleep(params["logging_timestep"])

        ## write the global stats
        if Agent.total_new_rcv_pkts > 0:
            loss_ratio = Agent.total_lost_pkts/Agent.total_new_rcv_pkts
        else:
            loss_ratio = -1
        if Agent.total_arrived_pkts > 0:
            avg_delay = Agent.total_rewards/Agent.total_arrived_pkts
            avg_cost = Agent.total_rewards_with_loss/Agent.total_arrived_pkts
            avg_hops = Agent.total_hops/Agent.total_arrived_pkts
        else:
            avg_delay = -1
            avg_cost = -1
            avg_hops = -1

        with summary_writer_session.as_default():
            tf.summary.scalar('total_rewards_over_iterations', Agent.total_rewards, step=Agent.currIt)
            tf.summary.scalar('total_rewards_with_loss_over_iterations', Agent.total_rewards_with_loss, step=Agent.currIt)
            tf.summary.scalar('loss_ratio_over_iterations', loss_ratio, step=Agent.currIt)
            tf.summary.scalar('total_hops_over_iterations', Agent.total_hops, step=Agent.currIt)
            tf.summary.scalar('avg_hops_over_iterations', avg_hops, step=Agent.currIt)
            tf.summary.scalar('nb_buffered_pkts_over_iterations', Agent.total_new_rcv_pkts-(Agent.total_arrived_pkts + Agent.total_lost_pkts), step=Agent.currIt)
            tf.summary.scalar('total_rewards_over_time', Agent.total_rewards, step=int(Agent.curr_time*1e6))
            tf.summary.scalar('total_rewards_with_loss_over_time', Agent.total_rewards_with_loss, step=int(Agent.curr_time*1e6))
            tf.summary.scalar('loss_ratio_over_time', loss_ratio, step=int(Agent.curr_time*1e6))
            tf.summary.scalar('total_hops_over_time', Agent.total_hops, step=int(Agent.curr_time*1e6))
            tf.summary.scalar('avg_hops_over_time', avg_hops, step=int(Agent.curr_time*1e6))
            tf.summary.scalar('nb_buffered_pkts_over_time', Agent.total_new_rcv_pkts-(Agent.total_arrived_pkts + Agent.total_lost_pkts), step=int(Agent.curr_time*1e6))
            tf.summary.scalar('total_arrived_pkts_over_iterations', Agent.total_arrived_pkts, step=Agent.currIt)
            tf.summary.scalar('total_arrived_pkts_over_time', Agent.total_arrived_pkts, step=int(Agent.curr_time*1e6))
            tf.summary.scalar('total_new_rcv_pkts_over_iterations', Agent.total_new_rcv_pkts, step=Agent.currIt)
            tf.summary.scalar('total_new_rcv_pkts_over_time', Agent.total_new_rcv_pkts, step=int(Agent.curr_time*1e6))
            tf.summary.scalar('total_lost_pkts_over_iterations', Agent.total_lost_pkts, step=Agent.currIt)
            tf.summary.scalar('total_lost_pkts_over_time', Agent.total_lost_pkts, step=int(Agent.curr_time*1e6))
            tf.summary.scalar('avg_cost_over_iterations', avg_cost, step=Agent.currIt)
            tf.summary.scalar('avg_cost_over_time', avg_cost, step=int(Agent.curr_time*1e6))
            tf.summary.scalar('avg_delay_over_iterations', avg_delay, step=Agent.currIt)
            tf.summary.scalar('avg_delay_over_time', avg_delay, step=int(Agent.curr_time*1e6))


        # print(f"""{Agent.currIt}, {Agent.total_rewards}, {Agent.total_new_rcv_pkts}, {Agent.total_arrived_pkts}, {Agent.total_lost_pkts}, {loss_ratio}, {avg_delay}""", file=global_log_file)
    print(f""" Summary of the episode :
            Total number of Transitions = {Agent.currIt}, 
            Total e2e delay = {Agent.total_rewards}, 
            Total number of packets = {Agent.total_new_rcv_pkts}, 
            Number of arrived packets = {Agent.total_arrived_pkts},
            Number of lost packets = {Agent.total_lost_pkts},
            Loss ratio = {loss_ratio},
            """)
    if Agent.total_arrived_pkts:
        print(f"Average delay per arrived packets = {Agent.total_rewards/Agent.total_arrived_pkts}")

if __name__ == '__main__':
    ## create a process group
    import traceback
    os.setpgrp()
    try:
        start_time = time()
        main()
        print("Elapsed time = ", str(datetime.timedelta(seconds= time() - start_time)))
    except Exception as e:
        traceback.print_exc()
    finally:
        os.killpg(0, signal.SIGKILL)