"""
This script will run all the experiments for the overlay 5 nodes full mesh topologies
For the traff matrix 0, we will vary the following  parameters:
    - exploration : fixed at 10% or variable from 100% to 1%
    - Frequency of sending the ping packets: [1, 10, 100]
    - Moving average interval : [1, 5, 10, 20]
    - Training load : [40%, 60%, 80%]
    - Batch size : [512, 1024]
    - Learning rate : [0.001, 0.0001]
    - State : [ with or without throughputs]
"""
import os

# static parameters
traff_mats = [0, 1, 2, 3]
sync_steps = list(range(10))
seeds = [100]
rb_size = 10000

experiment_name = "5_nodes_full_mesh_overlay_experiment_with_signalling"
starting_port = 700

# variable parameters
training_loads = [0.4, 0.6, 0.8]
mv_avg_intervals = [1, 10]
ping_freqs = [10, 100]
batch_sizes = [512]
learning_rates = [0.0001]
state_with_throughputs = [""]
explorations = [["fixed", 0.1, 0.1]]

inc = 0
for bs in batch_sizes:
    for traff_mat in traff_mats:
        for sync_step in sync_steps:
            for seed in seeds:
                for lr in learning_rates:
                    for exploration in explorations:
                        for state in state_with_throughputs:
                            for ping_freq in ping_freqs:
                                for mv_avg_interval in mv_avg_intervals:
                                    for trainin_load in training_loads:
                                        session_name = f"sync_{sync_step}_seed_{seed}_traff_mat_{traff_mat}_dqn_buffer{state}_NN_ping_freq_{ping_freq}_rb_size_{rb_size}_abilene_train_load_{trainin_load}_simTime_350_mv_avg_{mv_avg_interval}_lr_{lr}_bs_{bs}_explo_{exploration[0]}"
                                        os.system(f'tsp ./scripts/run_5_nodes.sh {sync_step} {seed} {traff_mat} dqn_buffer{state} NN {ping_freq} {rb_size} {inc} abilene 0 {trainin_load} 1 350 "{session_name}" {experiment_name} {mv_avg_interval} {lr} {bs} {exploration[1]} {exploration[2]}')
                                        inc +=1
print(inc)