#!/usr/bin python3
# -*- coding: utf-8 -*-
""" -----Main file for the PRISMA project----- """


__author__ = "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__version__ = "0.1.0"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"

### imports
from source.forwarder import Forwarder
from source.trainer import Trainer
from source.agent import Agent
from source.utils import save_model, save_all_models, convert_tb_data
from source.run_ns3 import run_ns3
from source.tb_logger import custom_plots, stats_writer_train, stats_writer_test
from source.argument_parser import parse_arguments
from source.utils import allocate_on_gpu, fix_seed
from time import sleep, time
import numpy as np
import threading
import copy
import tensorflow as tf
import os
import datetime
import json
from tensorboard.plugins.hparams import api as hp
import subprocess, signal
import shlex
import pathlib


def main():
    ## Allocate GPU memory as needed
    allocate_on_gpu()
    
    ## Get the arguments from the parser
    params = parse_arguments()

    ## fix the seed
    fix_seed(params["seed"])
    
    ## fill model version
    if "dqn_buffer" not in params["agent_type"] or params["train"] == 1:
        params["model_version"] = ""
    else:
        params["model_version"] = params["load_path"].split("/")[-1]
    
    ## check if the session already exists and the model is already trained
    if params["train"] == 1:
        pathlib.Path(params["logs_parent_folder"] + "/saved_models/").mkdir(parents=True, exist_ok=True)
        if os.path.exists(params["logs_parent_folder"] + "/saved_models/" + params["session_name"] + "/final"):
            if len(os.listdir(params["logs_parent_folder"] + "/saved_models/" + params["session_name"] + "/final")) > 0:
                print(f'The couple {params["seed"]} {params["traffic_matrix_index"]} already exists in : {params["logs_parent_folder"] + "/saved_models/" + params["session_name"]}')
                return None, None
    
    ## check if the test is already done   
    else:
        if os.path.exists(f"{params['logs_parent_folder']}/{params['session_name']}/test_results/{params['model_version']}"):
            if len(os.listdir(f"{params['logs_parent_folder']}/{params['session_name']}/test_results/{params['model_version']}")) > 0:
                ## check if the test load factor is already in the tensorboard file
                try: 
                    if int(100 * params["load_factor"]) in convert_tb_data(f"{params['logs_parent_folder']}/{params['session_name']}/test_results/{params['model_version']}")["step"].values:
                        print(f'The test session with load factor {params["load_factor"]} already exists in the {params["session_name"]}/test_results/{params["model_version"]} folder')
                        return None, None
                except:
                    pass
                            
    ## Setup writer for the global stats
    if params["train"] == 1:
        summary_writer_parent = tf.summary.create_file_writer(logdir=params["logs_folder"] )
        summary_writer_session = tf.summary.create_file_writer(logdir=params["global_stats_path"] )
        summary_writer_nb_arrived_pkts = tf.summary.create_file_writer(logdir=params["nb_arrived_pkts_path"] )
        summary_writer_nb_new_pkts = tf.summary.create_file_writer(logdir=params["nb_new_pkts_path"] )
        summary_writer_nb_lost_pkts = tf.summary.create_file_writer(logdir=params["nb_lost_pkts_path"] )

        ## write the session info (parameters)
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
    
    ## start the profiler
    if params["profile_session"]:
        from viztracer import VizTracer
        tracer = VizTracer(tracer_entries=5000000, min_duration=100, max_stack_depth=20, output_file=f"{params['logs_parent_folder'].rstrip('/')}/{params['session_name']}/viztracer.json")
        tracer.start()
    
    ## run ns3 simulator
    print("running ns-3")
    ns3_proc_id = run_ns3(params)
    ## run the agents threads
    for index in params["G"].nodes(): #range(params["numNodes"]):
        print("Starting agent", index)
        ## create the agent class instance
        forwarder_instance = Forwarder(index, agent_type=params["agent_type"], train=params["train"])
        ## start the agent forwarder thread
        th1 = threading.Thread(target=forwarder_instance.run, args=())
        th1.start()
        if params["train"]:
            trainer_instance = Trainer(index, agent_type=params["agent_type"], train=params["train"])
            ## start the agent trainer thread
            th2 = threading.Thread(target=trainer_instance.run, args=(), daemon=True)
            th2.start()

    ## Run tensorboard server
    tensorboard_process = None
    if params["start_tensorboard"]:
        args = shlex.split(f'python3 -m tensorboard.main --logdir={params["logs_folder"]} --port={params["tensorboard_port"]} --bind_all')
        tensorboard_process = subprocess.Popen(args).pid
        print(f"Tensorboard server started with pid {tensorboard_process}")
        
    sleep(1)
    
    snapshot_index = 1
    ## wait until simulation complete and update info about the env at each timestep
    while threading.active_count() > params["numNodes"] * (1+ params["train"]):
        sleep(params["logging_timestep"])
        if params["train"] == 1:
            stats_writer_train(summary_writer_session, summary_writer_nb_arrived_pkts, summary_writer_nb_lost_pkts, summary_writer_nb_new_pkts, Agent)

            # print(f"Time = {Agent.curr_time}, Overal injected packets = {Agent.sim_injected_packets}({Agent.total_new_rcv_pkts}), Overal delivered packets = {Agent.sim_delivered_packets}({Agent.total_arrived_pkts}), Overal lost packets = {Agent.sim_dropped_packets}({Agent.node_lost_pkts}), Overlay buffered packets = {Agent.sim_buffered_packets}({len(Agent.pkt_tracking_dict.keys())})")
            ## check if it is time to save a snapshot of the models
            if Agent.curr_time > (snapshot_index * params["snapshot_interval"]):
                print(f"Saving model at time {Agent.curr_time} with index {snapshot_index}")
                save_all_models(Agent.agents, params["G"].nodes(), params["session_name"], snapshot_index, 1, root=params["logs_parent_folder"] + "/saved_models/", snapshot=True)
                snapshot_index += 1
                    
    print(f""" Summary of the Simulation:
            Simulation time = {Agent.curr_time},
            Total Iterations = {Agent.total_nb_iterations},
            Total number of Transitions = {Agent.nb_transitions},
            Overlay Total injected packets = {Agent.sim_injected_packets}, 
            Global Total injected packets = {Agent.sim_global_injected_packets}, 
            Overlay arrived packets = {Agent.sim_delivered_packets},
            Global arrived packets = {Agent.sim_global_delivered_packets},
            Overlay lost packets = {Agent.sim_dropped_packets},
            Global lost packets = {Agent.sim_global_dropped_packets},
            Overlay buffered packets = {Agent.sim_buffered_packets},
            Global buffered packets = {Agent.sim_global_buffered_packets},
            Overlay lost ratio = {Agent.sim_dropped_packets/Agent.sim_injected_packets},
            Global lost ratio = {Agent.sim_global_dropped_packets/Agent.sim_global_injected_packets},
            Overlay e2e delay = {Agent.sim_avg_e2e_delay},
            Global e2e delay = {Agent.sim_global_avg_e2e_delay},
            Overlay Cost = {Agent.sim_cost},
            Global Cost = {Agent.sim_global_cost},
            Hops = {Agent.total_hops/Agent.sim_delivered_packets},
            # Overlay Data packet size = {Agent.sim_bytes_data},
            # Global Data packet size = {Agent.sim_global_bytes_data},
            # nbBytesBigSignaling = {Agent.sim_bytes_big_signaling},
            # nbBytesSmallSignaling = {Agent.sim_bytes_small_signaling},
            # nbBytesOverlaySignalingForward = {Agent.sim_bytes_overlay_signaling_forward},
            # nbBytesOverlaySignalingBack = {Agent.sim_bytes_overlay_signaling_back},
            OverheadRatio = {Agent.sim_signaling_overhead}
            """)                     
    
    ## write the results for the test session
    if params["train"] == 0:
       stats_writer_test(params["logs_folder"] + "/test_results", Agent)

    ## save models        
    if params["save_models"] and Agent.curr_time >= params["simTime"]-5:
        save_all_models(Agent.agents, params["G"].nodes(), params["session_name"], 1, 1, root=params["logs_parent_folder"] + "/saved_models/", snapshot=False)


    ## save the replay buffers
    if params["train"] == 1:
        for idxx, rb in enumerate(Agent.replay_buffer):
            if not os.path.exists(params["logs_folder"] + "/replay_buffers"):
                os.mkdir(path=params["logs_folder"] + "/replay_buffers")
            rb.save(params["logs_folder"] + "/replay_buffers/" + str(idxx) + ".pkl")
            
    ## save the profiler results
    if params["profile_session"]:
        tracer.stop()
        tracer.save() 
        
    return (ns3_proc_id, tensorboard_process)

if __name__ == '__main__':
    ## create a process group
    import traceback
    ns3_pid, tb_process = None, None
    # os.setpgrp()
    try:
        print("starting process group")
        start_time = time()
        ns3_pid, tb_process = main()
        print("Elapsed time = ", str(datetime.timedelta(seconds= time() - start_time)))
    except:
        traceback.print_exc()
        # write the error in the log file
        with open("examples/error.log", "a") as f:
            traceback.print_exc(file=f)
    finally:
        print("kill process group")
        if ns3_pid:
            os.system(command=f"kill -9 {ns3_pid}")
        if tb_process:
            os.system(command=f"kill -9 {tb_process}")
        SystemExit(0)
        # os.killpg(0, signal.SIGKILL)