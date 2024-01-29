
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### imports
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import os, subprocess
from time import sleep
import numpy as np
sys.path.append('../../source/')

__author__ = "Redha A. Alliche, Ramon Aparicio-Pardo, Lucile Sassatelli"
__copyright__ = "Copyright (c) 2023 Redha A. Alliche, Ramon Aparicio-Pardo, Lucile Sassatelli"
__license__ = "GPL"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr"

"""
This script will run all the experiments for the paper
We will vary the following  parameters:
    topology_name : ["abilene", # 5 nodes full mesh on abilene topology
                     "geant" # 10 nodes full mesh on geant topology
                     ]
    traffic matrix : [0-9]
    Target update period U when model sharing : [5, 7, 9]
    signaling_type : ["NN", # model sharing
                      "digital-twin"  # logit sharing
                      ]
    loss_penalty_type : ["fixed", # fixed loss penalty
                         "None",  # loss blind
                         "constrained" # guided reward
                         ]
    Experience relevancy thresholds : [0.0, 0.25, 0.5, 0.75]
"""
def generate_command(seed,
                     train,
                     sim_duration,
                     agent_type,
                     session_name,
                     traffic_matrix_index,
                     signaling_type, 
                     topology_name,
                     experiment_name,
                     batch_size,
                     save_models,
                     learning_rate,
                     exploration_initial_eps,
                     exploration_final_eps,
                     replay_buffer_max_size,
                     load_factor,
                     sync_step,
                     max_out_buffer_size,
                     movingAverageObsSize,
                     prioritizedReplayBuffer,
                     activateUnderlayTraffic,
                     bigSignalingSize,
                     loss_penalty_type,
                     smart_exploration,
                     load_path,
                     snapshot_interval,
                     lambda_train_step,
                     buffer_soft_limit,
                     lambda_lr,
                     lamda_training_start_time,
                     pingPacketIntervalTime,
                     d_t_max_time,
                     numEpisodes,
                     d_t_send_all_destinations,
                     rcpo_consider_loss,
                     reset_exploration,
                     rcpo_use_loss_pkts,
                     saved_models_path,
                     gap_threshold                                 
                  ):
    """ Generate the simulation command
    """
    simulation_command = f'python3 -u main.py --seed={seed} --simTime={sim_duration} --train={train} --basePort=7000 --agent_type={agent_type} --session_name={session_name} --signaling_type={signaling_type} --logs_parent_folder=examples/{topology_name}/results/{experiment_name} --traffic_matrix_root_path=examples/{topology_name}/traffic_matrices/ --traffic_matrix_index={traffic_matrix_index} --overlay_adjacency_matrix_path=examples/{topology_name}/topology_files/overlay_adjacency_matrix.txt --physical_adjacency_matrix_path=examples/{topology_name}/topology_files/physical_adjacency_matrix.txt --node_coordinates_path=examples/{topology_name}/topology_files/node_coordinates.txt --map_overlay_path=examples/{topology_name}/topology_files/map_overlay.txt --training_step=0.01 --batch_size={batch_size} --lr={learning_rate} --exploration_final_eps={exploration_final_eps} --exploration_initial_eps={exploration_initial_eps} --iterationNum=5000 --gamma=1.0 --save_models={save_models} --start_tensorboard=0 --replay_buffer_max_size={replay_buffer_max_size} --link_delay=1 --load_factor={load_factor} --sync_step={sync_step} --max_out_buffer_size={max_out_buffer_size} --sync_ratio=0.2 --signalingSim=1 --movingAverageObsSize={movingAverageObsSize} --prioritizedReplayBuffer={prioritizedReplayBuffer} --activateUnderlayTraffic={activateUnderlayTraffic} --bigSignalingSize={bigSignalingSize} --groundTruthFrequence=1 --pingAsObs=1 --load_path={load_path} --loss_penalty_type={loss_penalty_type} --snapshot_interval={snapshot_interval} --smart_exploration={smart_exploration} --lambda_train_step={lambda_train_step} --buffer_soft_limit={buffer_soft_limit} --lambda_lr={lambda_lr} --lamda_training_start_time={lamda_training_start_time} --d_t_max_time={d_t_max_time} --pingPacketIntervalTime={pingPacketIntervalTime} --numEpisodes={numEpisodes} --d_t_send_all_destinations={d_t_send_all_destinations} --rcpo_consider_loss={rcpo_consider_loss} --reset_exploration={reset_exploration} --rcpo_use_loss_pkts={rcpo_use_loss_pkts} --tunnels_max_delays_file_name=examples/{topology_name}/topology_files/max_observed_values.txt --saved_models_path={saved_models_path} --gap_threshold={gap_threshold} --packet_size=516'
    return simulation_command
    
 
# variable parameters
base_topo = "geant"
traff_mats = list(range(0, 10))
sync_steps = [5, 7, 9]
signaling_types= [ "NN", "digital_twin"]
loss_pen_types = [ "fixed", "None", "constrained"]
thresholds =  [0.0, 0.25, 0.5, 0.75]

# static parameters
train_load = 0.9 # load factor for training
seed = 100 # random seed
rb_size = 10000 # replay buffer size
nn_size = 35328 # size of the NN model
bs = 512 # batch size
lr = 0.00001 # learning rate
exploration = ["vary", 1.0, 0.01]  # exploration type, initial value, final value
mv_avg_interval = 5 # moving average window size
train_duration = 20 # training duration in seconds
max_output_buffer_size = 16260 # max output buffer size for each physical link
max_num_episodes = 20 # number of episodes
pingPacketIntervalTime = 0.1 # ping packet interval time
d_m_max_time = 3 # supervised learning timestamp for logit sharing
reset_exploration = 0 # reset exploration after each episode
test_loads = [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3] # load factors for testing
test_duration = 20 # testing duration in seconds
experiment_name = f"exps"
saved_models_path = f"examples/exps/saved_models/"

# get the results directory
os.chdir(f"../topologies/{base_topo}/results/")
# get the current working directory
results_wd = os.getcwd()

# go to the prisma directory
os.chdir("../../../../")

command_root = f"docker run --rm --gpus all -v {results_wd}:/app/prisma/examples -v /mnt/journal_paper_results/{base_topo}_overlay/saved_models:/mnt/journal_paper_results/{base_topo}_overlay/saved_models -w /app/prisma allicheredha/prismacopy_episodes:odqr"

for traff_mat in traff_mats:
    for loss_penalty_type in loss_pen_types:
        for signaling_type in signaling_types:
            if signaling_type == "NN":
                # model sharing
                for sync_step in sync_steps:
                    threshold = 0.0
                    session_name = f"sync_{sync_step}_mat_{traff_mat}_dqn__{signaling_type}_size_{nn_size}_tr_{train_load}_sim_{train_duration}_{max_num_episodes}_lr_{lr}_bs_{bs}_outb_{max_output_buffer_size}_losspen_{loss_penalty_type}_lambda_step_-1_ratio_0_wait_0_lambda_lr_1e6_dt_time_{d_m_max_time}_ping_{pingPacketIntervalTime}_{exploration[0]}_one_explo_first_loss_1_reset_{reset_exploration}_use_loss_1_threshold_0.0"
                    # launch training
                    python_command = generate_command(seed=seed,
                                                    train=1,
                                                    sim_duration=train_duration,
                                                    agent_type=f"dqn_buffer",
                                                    session_name=session_name,
                                                    traffic_matrix_index=traff_mat,
                                                    signaling_type=signaling_type, 
                                                    topology_name=base_topo,
                                                    experiment_name=experiment_name,
                                                    batch_size=bs,
                                                    learning_rate=lr,
                                                    save_models=1,
                                                    exploration_initial_eps=exploration[1],
                                                    exploration_final_eps=exploration[2],
                                                    replay_buffer_max_size=rb_size,
                                                    load_factor=train_load,
                                                    sync_step=sync_step,
                                                    max_out_buffer_size=max_output_buffer_size,
                                                    movingAverageObsSize=mv_avg_interval,
                                                    prioritizedReplayBuffer=0,
                                                    activateUnderlayTraffic=1,
                                                    bigSignalingSize=nn_size,
                                                    loss_penalty_type=loss_penalty_type,
                                                    snapshot_interval=train_duration,
                                                    smart_exploration=0,
                                                    load_path=f"examples/topologies/{base_topo}/pre_trained_models/dqn_buffer",
                                                    lambda_train_step=-1,
                                                    buffer_soft_limit=0,
                                                    lambda_lr=lr*0.1,
                                                    pingPacketIntervalTime=pingPacketIntervalTime,
                                                    lamda_training_start_time=0,
                                                    numEpisodes=max_num_episodes,
                                                    d_t_max_time=d_m_max_time,
                                                    d_t_send_all_destinations=0,
                                                    rcpo_consider_loss=1,
                                                    reset_exploration=reset_exploration,
                                                    rcpo_use_loss_pkts=1,
                                                    saved_models_path=saved_models_path,
                                                    gap_threshold=threshold
                                                    )
                    full_command = f'{command_root} {python_command}'
                    task_id = int(subprocess.check_output(full_command, shell=True))
                    ## launch testing
                    for test_load in test_loads:
                        for model_version in [f"episode_{i}_step_{i}" for i in [1, 2, 3, 4, 5, 6, 7, 10, 15, 19]] + ["final"]: # launch testing for the recorded models
                            python_command = generate_command(seed=seed,
                                                            train=0,
                                                            sim_duration=test_duration,
                                                            agent_type=f"dqn_buffer",
                                                            session_name=str(session_name),
                                                            traffic_matrix_index=traff_mat,
                                                            signaling_type="ideal", 
                                                            topology_name=base_topo,
                                                            experiment_name=experiment_name,
                                                            batch_size=bs,
                                                            learning_rate=lr,
                                                            save_models=0,
                                                            exploration_initial_eps=exploration[1],
                                                            exploration_final_eps=exploration[2],
                                                            replay_buffer_max_size=rb_size,
                                                            load_factor=test_load,
                                                            sync_step=sync_step,
                                                            max_out_buffer_size=max_output_buffer_size,
                                                            movingAverageObsSize=mv_avg_interval,
                                                            prioritizedReplayBuffer=0,
                                                            activateUnderlayTraffic=1,
                                                            bigSignalingSize=nn_size,
                                                            loss_penalty_type=loss_penalty_type,
                                                            snapshot_interval=0,
                                                            smart_exploration=0,
                                                            load_path=f"{saved_models_path}/{session_name}/{model_version}",
                                                            lambda_train_step=-1,
                                                            buffer_soft_limit=0,
                                                            lambda_lr=lr*0.1,
                                                            lamda_training_start_time=0,
                                                            pingPacketIntervalTime=pingPacketIntervalTime,
                                                            d_t_max_time=d_m_max_time,
                                                            numEpisodes=1,
                                                            d_t_send_all_destinations=0,
                                                            rcpo_consider_loss=1,
                                                            reset_exploration=reset_exploration,
                                                            rcpo_use_loss_pkts=1,
                                                            saved_models_path=saved_models_path,
                                                            gap_threshold=threshold
                                                            )
                            full_command = f'{command_root} {python_command}'
                            subprocess.check_output(full_command, shell=True)
            elif signaling_type == "digital_twin":
                # logit sharing
                for threshold in thresholds:
                    session_name = f"sync_1_mat_{traff_mat}_dqn__{signaling_type}_size_{nn_size}_tr_{train_load}_sim_{train_duration}_{max_num_episodes}_lr_{lr}_bs_{bs}_outb_{max_output_buffer_size}_losspen_{loss_penalty_type}_lambda_step_-1_ratio_0_wait_0_lambda_lr_1e6_dt_time_{d_m_max_time}_ping_{pingPacketIntervalTime}_{exploration[0]}_one_explo_first_loss_1_reset_{reset_exploration}_use_loss_1_threshold_{threshold}"
                    # launch training
                    python_command = generate_command(seed=seed,
                                                    train=1,
                                                    sim_duration=train_duration,
                                                    agent_type=f"dqn_buffer",
                                                    session_name=session_name,
                                                    traffic_matrix_index=traff_mat,
                                                    signaling_type=signaling_type, 
                                                    topology_name=base_topo,
                                                    experiment_name=experiment_name,
                                                    batch_size=bs,
                                                    learning_rate=lr,
                                                    save_models=1,
                                                    exploration_initial_eps=exploration[1],
                                                    exploration_final_eps=exploration[2],
                                                    replay_buffer_max_size=rb_size,
                                                    load_factor=train_load,
                                                    sync_step=1,
                                                    max_out_buffer_size=max_output_buffer_size,
                                                    movingAverageObsSize=mv_avg_interval,
                                                    prioritizedReplayBuffer=0,
                                                    activateUnderlayTraffic=1,
                                                    bigSignalingSize=nn_size,
                                                    loss_penalty_type=loss_penalty_type,
                                                    snapshot_interval=train_duration,
                                                    smart_exploration=0,
                                                    load_path=f"examples/topologies/{base_topo}/pre_trained_models/dqn_buffer",
                                                    lambda_train_step=-1,
                                                    buffer_soft_limit=0,
                                                    lambda_lr=lr*0.1,
                                                    pingPacketIntervalTime=pingPacketIntervalTime,
                                                    lamda_training_start_time=0,
                                                    numEpisodes=max_num_episodes,
                                                    d_t_max_time=d_m_max_time,
                                                    d_t_send_all_destinations=0,
                                                    rcpo_consider_loss=1,
                                                    reset_exploration=reset_exploration,
                                                    rcpo_use_loss_pkts=1,
                                                    saved_models_path=saved_models_path,
                                                    gap_threshold=threshold
                                                    )             
                    full_command = f'{command_root} {python_command}'
                    subprocess.check_output(full_command, shell=True)
                    ## launch testing
                    for test_load in test_loads:
                        for model_version in [f"episode_{i}_step_{i}" for i in [1, 2, 3, 4, 5, 6, 7, 10, 15, 19]] + ["final"]: # launch testing for the recorded models
                            python_command = generate_command(seed=seed,
                                                            train=0,
                                                            sim_duration=test_duration,
                                                            agent_type=f"dqn_buffer",
                                                            session_name=str(session_name),
                                                            traffic_matrix_index=traff_mat,
                                                            signaling_type="ideal", 
                                                            topology_name=base_topo,
                                                            experiment_name=experiment_name,
                                                            batch_size=bs,
                                                            learning_rate=lr,
                                                            save_models=0,
                                                            exploration_initial_eps=exploration[1],
                                                            exploration_final_eps=exploration[2],
                                                            replay_buffer_max_size=rb_size,
                                                            load_factor=test_load,
                                                            sync_step=1,
                                                            max_out_buffer_size=max_output_buffer_size,
                                                            movingAverageObsSize=mv_avg_interval,
                                                            prioritizedReplayBuffer=0,
                                                            activateUnderlayTraffic=1,
                                                            bigSignalingSize=nn_size,
                                                            loss_penalty_type=loss_penalty_type,
                                                            snapshot_interval=0,
                                                            smart_exploration=0,
                                                            load_path=f"{saved_models_path}/{session_name}/{model_version}",
                                                            lambda_train_step=-1,
                                                            buffer_soft_limit=0,
                                                            lambda_lr=lr*0.1,
                                                            lamda_training_start_time=0,
                                                            pingPacketIntervalTime=pingPacketIntervalTime,
                                                            d_t_max_time=d_m_max_time,
                                                            numEpisodes=1,
                                                            d_t_send_all_destinations=0,
                                                            rcpo_consider_loss=1,
                                                            reset_exploration=reset_exploration,
                                                            rcpo_use_loss_pkts=1,
                                                            saved_models_path=saved_models_path,
                                                            gap_threshold=threshold
                                                            )
                            full_command = f'{command_root} {python_command}'
                            subprocess.check_output(full_command, shell=True)
                                
    # launch testing for the shortest path routing and oracle routing
    for model in ["sp", "opt"]:
        for test_load in test_loads:
            session_name = f"{model}_{traff_mat}"
            python_command = generate_command(seed=seed,
                                            train=0,
                                            sim_duration=test_duration,
                                            agent_type=f"{model}",
                                            session_name=session_name,
                                            traffic_matrix_index=traff_mat,
                                            signaling_type="ideal", 
                                            topology_name=base_topo,
                                            experiment_name=experiment_name,
                                            batch_size=bs,
                                            learning_rate=lr,
                                            save_models=0,
                                            exploration_initial_eps=exploration[1],
                                            exploration_final_eps=exploration[2],
                                            replay_buffer_max_size=rb_size,
                                            load_factor=test_load,
                                            sync_step=1,
                                            max_out_buffer_size=max_output_buffer_size,
                                            movingAverageObsSize=mv_avg_interval,
                                            prioritizedReplayBuffer=0,
                                            activateUnderlayTraffic=1,
                                            bigSignalingSize=nn_size,
                                            loss_penalty_type=loss_penalty_type,
                                            snapshot_interval=0,
                                            smart_exploration=0,
                                            lambda_train_step=-1,
                                            buffer_soft_limit=0,
                                            lambda_lr=lr*0.1,
                                            lamda_training_start_time=0,
                                            pingPacketIntervalTime=pingPacketIntervalTime,
                                            d_t_max_time=d_m_max_time,
                                            numEpisodes=1,
                                            d_t_send_all_destinations=0,
                                            rcpo_consider_loss=1,
                                            reset_exploration=reset_exploration,
                                            rcpo_use_loss_pkts=1,
                                            saved_models_path=saved_models_path,
                                            gap_threshold=threshold
                                            )
            full_command = f'{command_root} {python_command}'
            print(full_command)
            subprocess.check_output(full_command, shell=True)

