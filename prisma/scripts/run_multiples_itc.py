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
                     movingAverageObsSize,
                     prioritizedReplayBuffer,
                     activateUnderlayTraffic,
                     bigSignalingSize,
                     d_t_max_time,
                     d_t_agent_type,
                     loss_penalty_type,
                     smart_exploration,
                     load_path,
                     d_t_load_path,
                     snapshot_interval,
                     lambda_train_step,
                     buffer_soft_limit_ratio,
                     lambda_lr,                     
                  ):
    """ Generate the simulation command
    """
    simulation_command = f'python3 main.py --seed={seed} --simTime={sim_duration} --train={train} --basePort=7000 --agent_type={agent_type} --session_name={session_name} --signaling_type={signaling_type} --logs_parent_folder=examples/{topology_name}/results/{experiment_name} --traffic_matrix_root_path=examples/{topology_name}/traffic_matrices/ --traffic_matrix_index={traffic_matrix_index} --overlay_adjacency_matrix_path=examples/{topology_name}/topology_files/overlay_adjacency_matrix.txt --physical_adjacency_matrix_path=examples/{topology_name}/topology_files/physical_adjacency_matrix.txt --node_coordinates_path=examples/{topology_name}/topology_files/node_coordinates.txt --map_overlay_path=examples/{topology_name}/topology_files/map_overlay.txt --training_step=0.01 --batch_size={batch_size} --lr={learning_rate} --exploration_final_eps={exploration_final_eps} --exploration_initial_eps={exploration_initial_eps} --iterationNum=5000 --gamma=1.0 --save_models={save_models} --start_tensorboard=0 --replay_buffer_max_size={replay_buffer_max_size} --link_delay="1ms" --load_factor={load_factor} --sync_step={sync_step} --max_out_buffer_size={max_out_buffer_size} --sync_ratio=0.2 --signalingSim=1 --movingAverageObsSize={movingAverageObsSize} --prioritizedReplayBuffer={prioritizedReplayBuffer} --activateUnderlayTraffic={activateUnderlayTraffic} --bigSignalingSize={bigSignalingSize} --groundTruthFrequence=1 --pingAsObs=0 --load_path={load_path} --d_t_load_path={d_t_load_path} --loss_penalty_type={loss_penalty_type} --snapshot_interval={snapshot_interval} --smart_exploration={smart_exploration} --lambda_train_step={lambda_train_step} --buffer_soft_limit_ratio={buffer_soft_limit_ratio} --lambda_lr={lambda_lr}'
    return simulation_command
    
 
import os, subprocess
from time import sleep
# static parameters
# traff_mats = [3, 2, 0, 1]
# traff_mats = [1, 0, 2, 3]
traff_mats = [0,]
# sync_steps = list(range(1, 7))
sync_steps = [1,]
seed = 100
rb_sizes = [10000,] 
# signaling_types = ["NN", "digital_twin", "target"]
signaling_types = ["NN",]
# signaling_types = ["digital_twin"]
dqn_models = ["", "_lite", "_lighter", "_lighter_2", "_lighter_3", "_ff"]
dqn_models = ["", "_lite", "_lighter", "_lighter_2", "_lighter_3"]
dqn_models = [""]
experiment_name = "ITC_NN_size_variations_experiment_rcpo"
nn_sizes = [35328, 9728, 5120, 1536, 512, 1024]
nn_sizes = [35328, 9728, 5120, 1536, 512]
nn_sizes = [35328]
d_t_max_time = 10
topology_name = "abilene"
# variable parameters
training_load = 0.4
bs = 1024
lr = 0.0001
# explorations = [["vary", 1.0, 0.01],
#                 ["fixed", 0.1, 0.1]]
explorations = [["vary", 1.0, 0.01]]
smart_explorations = [0,]
ping_freq = 10000
mv_avg_interval = 100
train_duration = 60
test_duration = 20
max_output_buffer_sizes = [16260,]
loss_pen_types = ["constrained"]
train_load = 0.4
test_loads = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
test_loads = []
buffer_soft_limit_ratios = [0.45,]
lambda_train_steps = [1,]


inc = 0
command_root = ""#"docker run --rm --gpus all -v /home/redha/PRISMA_copy/prisma/examples:/app/prisma/examples -w /app/prisma allicheredha/prismacopy:2.8.3"
for traff_mat in traff_mats:
    for rb_size in rb_sizes:
        for loss_penalty_type in loss_pen_types:
            for lambda_train_step in lambda_train_steps:
                for buffer_soft_limit_ratio in buffer_soft_limit_ratios:
                    if loss_penalty_type != "constrained":
                        if not(lambda_train_step == 1 and buffer_soft_limit_ratio == 0.6):
                            continue
                    for max_output_buffer_size in max_output_buffer_sizes:
                        for exploration in explorations:
                            for smart_exploration in smart_explorations:
                                for signaling_type in signaling_types:
                                    for idx, dqn_model in enumerate(dqn_models):
                                        if signaling_type == "NN":
                                            for sync_step in sync_steps:
                                                    session_name = f"sync_{sync_step}_traff_mat_{traff_mat}_dqn_buffer{dqn_model}_{signaling_type}_rb_size_{rb_size}_{topology_name}_train_load_{training_load}_simTime_{train_duration}_lr_{lr}_bs_{bs}_explo_{exploration[0]}_{['natural', 'smart'][smart_exploration]}_output_buffer_{max_output_buffer_size}_loss_pen_{loss_penalty_type}_lambda_step_{lambda_train_step}_soft_limit_ratio_{buffer_soft_limit_ratio}_lr_1e6"
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
                                                                                    d_t_max_time=d_t_max_time,
                                                                                    d_t_agent_type=f"dqn_buffer{dqn_model}",
                                                                                    loss_penalty_type=loss_penalty_type,
                                                                                    snapshot_interval=10,
                                                                                    smart_exploration=smart_exploration,
                                                                                    load_path=f"examples/{topology_name}/pre_trained_models/dqn_buffer{dqn_model}",
                                                                                    d_t_load_path=f"examples/{topology_name}/pre_trained_models/dqn_buffer{dqn_model}",
                                                                                    lambda_train_step=lambda_train_step,
                                                                                    buffer_soft_limit_ratio=buffer_soft_limit_ratio,
                                                                                    lambda_lr=0.000001,
                                                                                    )
                                                                            
                                                    full_command = f'{python_command}'
                                                    # print("vs code config args = ", sum([x.split("=") for x in python_command.split(" ")[3:]], []))
                                                    task_id = int(subprocess.run(full_command))
                                                    print("train", full_command)
                                                    sleep(1)
                                                    # put the job on the top of the queue
                                                    # subprocess.check_output(f"tsp -u {task_id}", shell=True)
                                                    # launch testing for final model and intermediate models
                                                    saved_models_path = f"examples/{topology_name}/results/{experiment_name}/saved_models/{session_name}"
                                                    for test_load in test_loads:
                                                        for model_version in ["final"] + [f"episode1_step_{i}" for i in range(1, 6)]:
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
                                                                                            d_t_max_time=d_t_max_time,
                                                                                            d_t_agent_type=f"dqn_buffer{dqn_model}",
                                                                                            loss_penalty_type=loss_penalty_type,
                                                                                            snapshot_interval=0,
                                                                                            smart_exploration=smart_exploration,
                                                                                            load_path=f"{saved_models_path}/{model_version}",
                                                                                            d_t_load_path=f"{saved_models_path}/{model_version}",
                                                                                            lambda_train_step=lambda_train_step,
                                                                                            buffer_soft_limit_ratio=buffer_soft_limit_ratio,
                                                                                            lambda_lr=0.000001,
                                                                                            
                                                                                            )
                                                            full_command = f'tsp -D {task_id} {command_root} {python_command}'
                                                            test_task_id = int(subprocess.check_output(full_command, shell=True))
                                                            print(task_id, "test", test_load)
                                                            # put the job on the top of the queue
                                                            # subprocess.check_output(f"tsp -u {test_task_id}", shell=True)

                                                            sleep(2)
                                                    
                                                        
                                        else:
                                            sync_step = 1
                                            session_name = f"sync_{sync_step}_seed_{seed}_traff_mat_{traff_mat}_dqn_buffer{dqn_model}_{signaling_type}_ping_freq_{ping_freq}_rb_size_{rb_size}_{topology_name}_train_load_{training_load}_simTime_{train_duration}_mv_avg_{mv_avg_interval}_lr_{lr}_bs_{bs}_explo_{exploration[0]}_{d_t_max_time}_{nn_sizes[idx]}"
                                            os.system(f'tsp {command_root} ./scripts/run_geant.sh {sync_step} {seed} {traff_mat} dqn_buffer{dqn_model} {signaling_type} {ping_freq} {rb_size} {inc} {topology_name} 0 {training_load} 1 {train_duration} "{session_name}" {experiment_name} {mv_avg_interval} {lr} {bs} {exploration[1]} {exploration[2]} {d_t_max_time} {nn_sizes[idx]} dqn_buffer{dqn_model}')
                                            sleep(0.5)
                                            inc +=1
    # print(inc)
