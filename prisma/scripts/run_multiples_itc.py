"""
This script will run all the experiments for ITC {topology_name} 11 nodes topologie
We will vary the following  parameters:
    traffic matrix : [0, 1, 2, 3]
    sync step : [1, 2, 3, 4, 5, 6, 7, 8, 9]
    signaling_type : ["NN", "digital-twin", "target"]
    dqn model : ["original", "light", "lighter", "lighter_2", "lighter_3", "ff"]
    
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
                     nPacketsOverlay,
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
                     numEpisodes
                                   
                  ):
    """ Generate the simulation command
    """
    simulation_command = f'python3 -u main.py --seed={seed} --simTime={sim_duration} --train={train} --basePort=7000 --agent_type={agent_type} --session_name={session_name} --signaling_type={signaling_type} --logs_parent_folder=examples/{topology_name}/results/{experiment_name} --traffic_matrix_root_path=examples/{topology_name}/traffic_matrices/ --traffic_matrix_index={traffic_matrix_index} --overlay_adjacency_matrix_path=examples/{topology_name}/topology_files/overlay_adjacency_matrix.txt --physical_adjacency_matrix_path=examples/{topology_name}/topology_files/physical_adjacency_matrix.txt --node_coordinates_path=examples/{topology_name}/topology_files/node_coordinates.txt --map_overlay_path=examples/{topology_name}/topology_files/map_overlay.txt --training_step=0.01 --batch_size={batch_size} --lr={learning_rate} --exploration_final_eps={exploration_final_eps} --exploration_initial_eps={exploration_initial_eps} --iterationNum=5000 --gamma=1.0 --save_models={save_models} --start_tensorboard=0 --replay_buffer_max_size={replay_buffer_max_size} --link_delay="1ms" --load_factor={load_factor} --sync_step={sync_step} --max_out_buffer_size={max_out_buffer_size} --sync_ratio=0.2 --signalingSim=1 --nPacketsOverlay={nPacketsOverlay} --movingAverageObsSize={movingAverageObsSize} --prioritizedReplayBuffer={prioritizedReplayBuffer} --activateUnderlayTraffic={activateUnderlayTraffic} --bigSignalingSize={bigSignalingSize} --groundTruthFrequence=1 --pingAsObs=1 --load_path={load_path} --loss_penalty_type={loss_penalty_type} --snapshot_interval={snapshot_interval} --smart_exploration={smart_exploration} --lambda_train_step={lambda_train_step} --buffer_soft_limit={buffer_soft_limit} --lambda_lr={lambda_lr} --lamda_training_start_time={lamda_training_start_time} --d_t_max_time={d_t_max_time} --pingPacketIntervalTime={pingPacketIntervalTime} --numEpisodes={numEpisodes}'
    return simulation_command
    
 
import os, subprocess
from time import sleep
# static parameters
# traff_mats = [3, 2, 0, 1]
traff_mats = [0,]
# traff_mats = [0,]
sync_steps = list((3, 6, 10))
sync_steps = [1,2, 4, 6, 8 ,10]
sync_steps = [2, 4, 6]
sync_steps = [2, 4,]
seed = 100
rb_sizes = [10000,] 
# signaling_types = ["NN", "digital_twin", "target"]
signaling_types = ["ideal", "digital_twin", "NN"]
signaling_types = ["NN"]
# signaling_types = ["ideal"]
dqn_models = ["", "_lite", "_lighter", "_lighter_2", "_lighter_3", "_ff"]
dqn_models = ["", "_lite", "_lighter", "_lighter_2", "_lighter_3"]
dqn_models = ["sp", "opt"]
dqn_models = [""]
experiment_name = "ITC_NN_size_variations_experiment_rcpo_final_05_june_rcpo"
nn_sizes = [35328, 9728, 5120, 1536, 512, 1024]
nn_sizes = [35328, 9728, 5120, 1536, 512]
nn_sizes = [35328,]
# d_t_max_time = 10
topology_name = "5n_overlay_full_mesh_abilene"
# variable parameters
training_load = 0.4
bs = 512
lr = 0.00001
explorations = [["vary", 1.0, 0.01],
                ["fixed", 0.1, 0.1]]
# explorations = [["vary", 1.0, 0.01]]
smart_explorations = [0,]
ping_freq = 5
mv_avg_interval = 5
train_duration = 15
test_duration = 25
max_output_buffer_sizes = [16260,]
loss_pen_types = ["fixed", "constrained"]
loss_pen_types = ["constrained"]
lambda_waits = [0, 25, 80]
lambda_waits = [0, 80, train_duration]
lambda_waits = [0, 40]
lambda_waits = [0]
train_load = 0.4
test_loads = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
buffer_soft_limits = [0.6, 0.4, 0.1]
buffer_soft_limits = [0.6,]
lambda_train_steps = [1, 5, 10]
lambda_train_steps = [1,]
max_num_episodes = 30
d_t_max_times = [1, 5, 10]
pingPacketIntervalTimes = [0.1,]
# [0.16, 0.05, 0.03, 0.01]
d_t_max_times = [1, 5, 10]

inc = 0
command_root = "docker run --rm --gpus all -v /home/redha/PRISMA_copy/prisma/examples:/app/prisma/examples -w /app/prisma allicheredha/prismacopy_episodes"
for traff_mat in traff_mats:
    for lambda_wait in lambda_waits:
        for rb_size in rb_sizes:
            for loss_penalty_type in loss_pen_types:
                for lambda_train_step in lambda_train_steps:
                    for pingPacketIntervalTime in pingPacketIntervalTimes:
                        for buffer_soft_limit in buffer_soft_limits:
                            if loss_penalty_type != "constrained":
                                if not(lambda_train_step == 1 and buffer_soft_limit == 0.6 and lambda_wait==0):
                                    continue
                            else:
                                if lambda_wait == 100 and (buffer_soft_limit!=0.6):
                                    continue
                            for max_output_buffer_size in max_output_buffer_sizes:
                                for exploration in explorations:
                                    for smart_exploration in smart_explorations:
                                        for signaling_type in signaling_types:
                                            if signaling_type == "NN" or signaling_type == "digital_twin":
                                                for idx, dqn_model in enumerate(dqn_models):
                                                    for sync_step in sync_steps:
                                                        for d_m_max_time in d_t_max_times:
                                                            if d_m_max_time != 1 and signaling_type == "NN":
                                                                continue
                                                            if sync_step != 1 and signaling_type == "digital_twin":
                                                                continue
                                                            
                                                            session_name = f"sync_{sync_step}_mat_{traff_mat}_dqn_{dqn_model}_{signaling_type}_size_{nn_sizes[idx]}_{topology_name}_tr_{training_load}_sim_{train_duration}_lr_{lr}_bs_{bs}_outb_{max_output_buffer_size}_losspen_{loss_penalty_type}_lambda_step_{lambda_train_step}_soft_ratio_{buffer_soft_limit}_wait_{lambda_wait}_lambda_lr_1e7_dt_time_{d_m_max_time}_ping_{pingPacketIntervalTime}_{exploration[0]}"
                                                            # launch training
                                                            python_command = generate_command(seed=seed,
                                                                                            train=1,
                                                                                            sim_duration=train_duration,
                                                                                            agent_type=f"dqn_buffer{dqn_model}",
                                                                                            session_name=session_name,
                                                                                            traffic_matrix_index=traff_mat,
                                                                                            signaling_type=signaling_type, 
                                                                                            topology_name=topology_name,
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
                                                                                            nPacketsOverlay=ping_freq,
                                                                                            movingAverageObsSize=mv_avg_interval,
                                                                                            prioritizedReplayBuffer=0,
                                                                                            activateUnderlayTraffic=1,
                                                                                            bigSignalingSize=nn_sizes[idx],
                                                                                            loss_penalty_type=loss_penalty_type,
                                                                                            snapshot_interval=10,
                                                                                            smart_exploration=smart_exploration,
                                                                                            load_path=f"examples/{topology_name}/pre_trained_models/dqn_buffer{dqn_model}",
                                                                                            lambda_train_step=lambda_train_step,
                                                                                            buffer_soft_limit=buffer_soft_limit,
                                                                                            lambda_lr=lr*0.01,
                                                                                            pingPacketIntervalTime=pingPacketIntervalTime,
                                                                                            lamda_training_start_time=lambda_wait,
                                                                                            numEpisodes=max_num_episodes,
                                                                                            d_t_max_time=d_m_max_time,
                                                                                            )
                                                                                    
                                                            full_command = f'tsp {command_root} {python_command}'
                                                            print(full_command)
                                                            # raise(0)
                                                            # print("vs code config args = ", sum([x.split("=") for x in python_command.split(" ")[3:]], []))
                                                            # task_id = int(subprocess.check_output(full_command, shell=True))
                                                            task_id = 0
                                                            print(task_id, "train", full_command)
                                                            sleep(0.3)
                                                            # put the job on the top of the queue
                                                            # subprocess.check_output(f"tsp -u {task_id}", shell=True)
                                                            # launch testing for final model and intermediate models
                                                            saved_models_path = f"examples/{topology_name}/results/{experiment_name}/saved_models/{session_name}"
                                                            for test_load in test_loads:
                                                                # for model_version in ["final"] + [f"episode_1_step_{i}" for i in range(2, 10, 4)]:
                                                                for model_version in ["episode_1_step_15"]:
                                                                    python_command = generate_command(seed=seed,
                                                                                                    train=0,
                                                                                                    sim_duration=test_duration,
                                                                                                    agent_type=f"dqn_buffer{dqn_model}",
                                                                                                    session_name=session_name,
                                                                                                    traffic_matrix_index=traff_mat,
                                                                                                    signaling_type="ideal", 
                                                                                                    topology_name=topology_name,
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
                                                                                                    nPacketsOverlay=ping_freq,
                                                                                                    movingAverageObsSize=mv_avg_interval,
                                                                                                    prioritizedReplayBuffer=0,
                                                                                                    activateUnderlayTraffic=1,
                                                                                                    bigSignalingSize=nn_sizes[idx],
                                                                                                    loss_penalty_type=loss_penalty_type,
                                                                                                    snapshot_interval=0,
                                                                                                    smart_exploration=smart_exploration,
                                                                                                    load_path=f"{saved_models_path}/{model_version}",
                                                                                                    lambda_train_step=lambda_train_step,
                                                                                                    buffer_soft_limit=buffer_soft_limit,
                                                                                                    lambda_lr=lr*0.01,
                                                                                                    lamda_training_start_time=lambda_wait,
                                                                                                    pingPacketIntervalTime=pingPacketIntervalTime,
                                                                                                    d_t_max_time=d_m_max_time,
                                                                                                    numEpisodes=1,
                                                                                                    )
                                                                    full_command = f'tsp {command_root} {python_command}'
                                                                    sleep(0.3)
                                                                    # raise(0)
                                                                    print(full_command)
                                                                    test_task_id = int(subprocess.check_output(full_command, shell=True))
                                                                    print(task_id, "test", test_load)
                                                                    # put the job on the top of the queue
                                                                    # subprocess.check_output(f"tsp -u {test_task_id}", shell=True)

                                                                    # sleep(2)
                                                            
                                                                
                                            else:
                                                if signaling_type == "ideal":
                                                    for model in ["sp", "opt"]:
                                                        sync_step = 1
                                                        for test_load in test_loads:
                                                            python_command = generate_command(seed=seed,
                                                                                            train=0,
                                                                                            sim_duration=test_duration,
                                                                                            agent_type=f"{model}",
                                                                                            session_name=f"{model}_{traff_mat}_{test_duration}_final_ping_{pingPacketIntervalTime}",
                                                                                            traffic_matrix_index=traff_mat,
                                                                                            signaling_type="ideal", 
                                                                                            topology_name=topology_name,
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
                                                                                            nPacketsOverlay=ping_freq,
                                                                                            movingAverageObsSize=mv_avg_interval,
                                                                                            prioritizedReplayBuffer=0,
                                                                                            activateUnderlayTraffic=1,
                                                                                            bigSignalingSize=nn_sizes[0],
                                                                                            loss_penalty_type=loss_penalty_type,
                                                                                            snapshot_interval=0,
                                                                                            smart_exploration=smart_exploration,
                                                                                            load_path=None,
                                                                                            lambda_train_step=lambda_train_step,
                                                                                            buffer_soft_limit=buffer_soft_limit,
                                                                                            lambda_lr=lr*0.01,
                                                                                            lamda_training_start_time=lambda_wait,
                                                                                            pingPacketIntervalTime=pingPacketIntervalTime,
                                                                                            d_t_max_time=10,
                                                                                            numEpisodes=1,
                                                                                            )
                                                            full_command = f'tsp {command_root} {python_command}'
                                                            print(full_command)
                                                            test_task_id = int(subprocess.check_output(full_command, shell=True))
                                                            print(test_task_id, "test", test_load)
                                                inc +=1
    # print(inc)
