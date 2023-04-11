"""
This script will run all the experiments for ITC {topology_name} 11 nodes topologie
We will vary the following  parameters:
    traffic matrix : [0, 1, 2, 3]
    sync step : [1, 2, 3, 4, 5, 6, 7, 8, 9]
    signaling_type : ["NN", "digital-twin", "target"]
    dqn model : ["original", "light", "lighter", "lighter_2", "lighter_3", "ff"]
    
"""
import os
from time import sleep
# static parameters
# traff_mats = [3, 2, 0, 1]
traff_mats = [1, 0, 2, 3]
# traff_mats = [0,]
sync_steps = list(range(1, 7))
# sync_steps = [1,2]
seed = 100
rb_sizes = [10000,] 
# signaling_types = ["NN", "digital_twin", "target"]
signaling_types = ["NN",]
# signaling_types = ["digital_twin"]
dqn_models = ["", "_lite", "_lighter", "_lighter_2", "_lighter_3", "_ff"]
dqn_models = ["", "_lite", "_lighter", "_lighter_2", "_lighter_3"]
dqn_models = [""]
experiment_name = "ITC_NN_size_variations_experiment"
starting_port = 700
nn_sizes = [35328, 9728, 5120, 1536, 512, 1024]
nn_sizes = [35328, 9728, 5120, 1536, 512]
nn_sizes = [35328]
d_t_max_time = 10
topology_name = "geant"
# variable parameters
training_load = 0.4
bs = 1024
lr = 0.0001
explorations = [["vary", 1.0, 0.01],]
smart_explorations = [0,]
ping_freq = 10000
mv_avg_interval = 100
train_duration = 60
max_output_buffer_sizes = [16260,]
loss_aware_values = [1,]

inc = 0
command_name = "docker run --rm --gpus all -v /home/redha/PRISMA_copy/prisma/examples:/app/prisma/examples -w /app/prisma allicheredha/prismacopy:2.8.2"
for traff_mat in traff_mats:
    for rb_size in rb_sizes:
        for loss_aware in loss_aware_values:
            for max_output_buffer_size in max_output_buffer_sizes:
                for exploration in explorations:
                    for smart_exploration in smart_explorations:
                        for signaling_type in signaling_types:
                            for idx, dqn_model in enumerate(dqn_models):
                                if signaling_type == "NN":
                                    for sync_step in sync_steps:
                                            session_name = f"sync_{sync_step}_seed_{seed}_traff_mat_{traff_mat}_dqn_buffer{dqn_model}_{signaling_type}_ping_freq_{ping_freq}_rb_size_{rb_size}_{topology_name}_train_load_{training_load}_simTime_{train_duration}_mv_avg_{mv_avg_interval}_lr_{lr}_bs_{bs}_explo_{exploration[0]}_{['natural', 'smart'][smart_exploration]}_{d_t_max_time}_{nn_sizes[idx]}_output_buffer_{max_output_buffer_size}_loss_aware_{loss_aware}"
                                            command = f'tsp {command_name} ./scripts/run_geant.sh {sync_step} {seed} {traff_mat} dqn_buffer{dqn_model} {signaling_type} {ping_freq} {rb_size} {inc} {topology_name} 0 {training_load} 1 {train_duration} "{session_name}" {experiment_name} {mv_avg_interval} {lr} {bs} {exploration[1]} {exploration[2]} {d_t_max_time} {nn_sizes[idx]} dqn_buffer{dqn_model} {smart_exploration} {loss_aware} {max_output_buffer_size}'
                                            print(command)
                                            os.system(command)
                                            sleep(2)
                                            inc +=1
                                else:
                                    sync_step = 1
                                    session_name = f"sync_{sync_step}_seed_{seed}_traff_mat_{traff_mat}_dqn_buffer{dqn_model}_{signaling_type}_ping_freq_{ping_freq}_rb_size_{rb_size}_{topology_name}_train_load_{training_load}_simTime_{train_duration}_mv_avg_{mv_avg_interval}_lr_{lr}_bs_{bs}_explo_{exploration[0]}_{d_t_max_time}_{nn_sizes[idx]}"
                                    os.system(f'tsp {command_name} ./scripts/run_geant.sh {sync_step} {seed} {traff_mat} dqn_buffer{dqn_model} {signaling_type} {ping_freq} {rb_size} {inc} {topology_name} 0 {training_load} 1 {train_duration} "{session_name}" {experiment_name} {mv_avg_interval} {lr} {bs} {exploration[1]} {exploration[2]} {d_t_max_time} {nn_sizes[idx]} dqn_buffer{dqn_model}')
                                    sleep(0.5)
                                    inc +=1
# print(inc)
