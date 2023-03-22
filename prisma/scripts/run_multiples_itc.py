"""
This script will run all the experiments for ITC abilene 11 nodes topologie
We will vary the following  parameters:
    traffic matrix : [0, 1, 2, 3]
    sync step : [1, 2, 3, 4, 5, 6, 7, 8, 9]
    signaling_type : ["NN", "digital-twin", "target"]
    dqn model : ["original", "light", "lighter", "lighter_2", "lighter_3", "ff"]
    
"""
import os

# static parameters
traff_mats = [0]
traff_mats = [0, 1, 3, 2]
sync_steps = list(range(1, 10))
seed = 100
rb_size = 10000
signaling_types = ["NN", "digital_twin", "target"]
# signaling_types = ["NN"]
dqn_models = ["", "_lite", "_lighter", "_lighter_2", "_lighter_3", "_ff"]
# dqn_models = ["_lighter", "_lighter_2", "_lighter_3"]
experiment_name = "ITC_NN_size_variations_experiment"
starting_port = 700
nn_sizes = [35328, 9728, 5120, 1536, 512, 1024]
# nn_sizes = [5120, 1536, 512]
d_t_max_time = 5
# variable parameters
training_load = 0.4
bs = 512
lr = 0.001
exploration = ["vary", 1.0, 0.01]
ping_freq = 10000
mv_avg_interval = 100
train_duration = 60

inc = 0
for traff_mat in traff_mats:
    for signaling_type in signaling_types:
        for idx, dqn_model in enumerate(dqn_models):
            if signaling_type == "NN":
                for sync_step in sync_steps:
                        session_name = f"sync_{sync_step}_seed_{seed}_traff_mat_{traff_mat}_dqn_buffer{dqn_model}_{signaling_type}_ping_freq_{ping_freq}_rb_size_{rb_size}_abilene_train_load_{training_load}_simTime_{100}_mv_avg_{mv_avg_interval}_lr_{lr}_bs_{1024}_explo_{exploration[0]}_{d_t_max_time}_{nn_sizes[idx]}"
                        os.system(f'tsp ./scripts/run_11_nodes.sh {sync_step} {seed} {traff_mat} dqn_buffer{dqn_model} {signaling_type} {ping_freq} {rb_size} {inc} abilene 0 {training_load} 1 {train_duration} "{session_name}" {experiment_name} {mv_avg_interval} {lr} {bs} {exploration[1]} {exploration[2]} {d_t_max_time} {nn_sizes[idx]}')
                        inc +=1
            else:
                sync_step = 1
                session_name = f"sync_{sync_step}_seed_{seed}_traff_mat_{traff_mat}_dqn_buffer{dqn_model}_{signaling_type}_ping_freq_{ping_freq}_rb_size_{rb_size}_abilene_train_load_{training_load}_simTime_{100}_mv_avg_{mv_avg_interval}_lr_{lr}_bs_{1024}_explo_{exploration[0]}_{d_t_max_time}_{nn_sizes[idx]}"
                os.system(f'tsp ./scripts/run_11_nodes.sh {sync_step} {seed} {traff_mat} dqn_buffer{dqn_model} {signaling_type} {ping_freq} {rb_size} {inc} abilene 0 {training_load} 1 {train_duration} "{session_name}" {experiment_name} {mv_avg_interval} {lr} {bs} {exploration[1]} {exploration[2]} {d_t_max_time} {nn_sizes[idx]}')
                inc +=1
print(inc)
