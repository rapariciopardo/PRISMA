"""
This script plots the results of a test session.
It will go through all the test files in a given directory and plot: 
    - The average cost for each sync step (cost vs. sync step)
    - The training overhead for each sync step (training overhead vs. sync step)
    - The average cost and overhead for each sync step (cost vs. overhead)
    - The average cost for each load (cost vs. load)
    - The end-to-end delay for each load (delay vs. load)
    - The loss rate for each load (loss rate vs. load)
It will retrive the information from the tensorboard files.
It will compare the models with shortest path routing and optimal routing as benchmark.

author: allicheredha@gmail.com
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append('source/')
from utils import convert_tb_data

def plot_metric_vs_sync_step(result_df, exps_name, metric_name, axis):
    # plot the metric vs. sync step given a dataframes of results
    data = result_df[result_df.name == metric_name].sort_values("step")
    x = data.step.values
    y = data.value.values
    axis.plot(x, y, label=exps_name)
    
    

# The directory where the test results are stored
dir_path = '/home/redha/PRISMA_copy/prisma/examples/abilene/ITC_NN_size_variations_experiment'

plt.rcParams.update({'font.family': "serif"})
plt.rcParams.update({'font.size': 26})
# evaluation metric name
metric_names = ["global_cost", "overhead", "global_delay", "loss rate"]
metric_name = "global_cost"
all_exps_results = []

exps_names = []
# static parameters
traff_mats = [0,]
sync_steps = list(range(1, 10))
seed = 100
rb_size = 10000
signaling_types = ["NN", "digital-twin", "target"]
signaling_types = ["NN", "sp", "opt", "digital-twin"]
dqn_models = ["", "_lite", "_lighter", "_lighter_2", "_lighter_3", "_ff"]
experiment_name = "ITC_NN_size_variations_experiment"
starting_port = 700
nn_sizes = [35328, 9728, 5120, 1536, 512, 1024]
d_t_max_time = 5
# variable parameters
training_load = 0.4
bs = 1024
lr = 0.001
exploration = ["vary", 1.0, 0.01]
ping_freq = 10000
mv_avg_interval = 100
train_duration = 100
#%% cost vs sync step
test_loads = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

all_exps_results = {}
# go through all the experiments
not_found = []
found = []
# create figure and axis
fig, ax = plt.subplots(figsize=(19.4, 10))
for signaling_type in signaling_types:
    for idx, dqn_model in enumerate(dqn_models):
        accu = []
        for traff_mat in traff_mats:
            for sync_step in sync_steps:
                if signaling_type != "NN":
                    if sync_step != 1:
                        continue 
                if signaling_type == "digital-twin" and sync_step == 1:
                    continue
                session_name = f"sync_{sync_step}_seed_{seed}_traff_mat_{traff_mat}_dqn_buffer{dqn_model}_{signaling_type}_ping_freq_{ping_freq}_rb_size_{rb_size}_abilene_train_load_{training_load}_simTime_{train_duration}_mv_avg_{mv_avg_interval}_lr_{lr}_bs_{bs}_explo_{exploration[0]}_{d_t_max_time}_{nn_sizes[idx]}"
                if signaling_type == "sp":
                    session_name = "sp"
                if signaling_type == "opt":
                    session_name = "opt"
                exps_names.append(session_name)
                
                path = f"{dir_path}/{session_name}/test_results"
                try :
                    if not os.path.exists(path):
                        raise(Exception("test results not found"))
                    # Get the results for the experiment
                    exp_df = convert_tb_data(path)
                    data = exp_df[exp_df.name == metric_name].sort_values("step")
                    accu.append(exp_df.value.values.mean())
                    # append the results to the dataframe
                    # plot_metric_vs_sync_step(exp_df, exp_name, "test_global_cost", ax)
                    # assert len(exp_df.step.unique()) != len(test_loads) # check if all the test loads are present
                except Exception as e:
                    print(session_name)
                    not_found.append(session_name)
                    continue
                found.append(session_name)
        if len(accu) == 1:
            accu = accu * len(sync_steps)
        if len(accu) != len(sync_steps):
            raise(Exception("data not found for " + signaling_type + " " + dqn_model))
        if signaling_type == "sp" and dqn_model != "":
            continue
        if signaling_type == "opt" and dqn_model != "":
            continue
        if signaling_type in ["sp", "opt"]:
            ax.plot(sync_steps, np.array(accu), label=f"{signaling_type}", linestyle="--", marker="o")
        else:
            ax.plot(sync_steps, np.array(accu), label=f"{signaling_type} {dqn_model}", marker="o")
        
plt.xlabel(xlabel="Sync step")
plt.ylabel(ylabel="Average cost")
plt.tight_layout()
plt.legend()
plt.savefig(f"cost_vs_sync_steps.png") 

#%% overhead vs sync step
signaling_types = ["NN", "target",]
metric_name = "overhead"
all_exps_results = {}
# go through all the experiments
not_found = []
found = []
# create figure and axis
fig, ax = plt.subplots(figsize=(19.4, 10))
for signaling_type in signaling_types:
    for idx, dqn_model in enumerate(dqn_models):
        accu = []
        for traff_mat in traff_mats:
            for sync_step in sync_steps:
                if signaling_type != "NN":
                    if sync_step != 1:
                        continue 
                if sync_step in (6, 7, 8, 9) and signaling_type == "NN":
                    session_name = f"sync_{sync_step}_seed_{seed}_traff_mat_{traff_mat}_dqn_buffer{dqn_model}_{signaling_type}_ping_freq_{ping_freq}_rb_size_{rb_size}_abilene_train_load_{training_load}_simTime_{100}_mv_avg_{mv_avg_interval}_lr_{lr}_bs_{bs}_explo_{exploration[0]}_{d_t_max_time}_{nn_sizes[idx]}"
                else :
                    session_name = f"sync_{sync_step}_seed_{seed}_traff_mat_{traff_mat}_dqn_buffer{dqn_model}_{signaling_type}_ping_freq_{ping_freq}_rb_size_{rb_size}_abilene_train_load_{training_load}_simTime_{train_duration}_mv_avg_{mv_avg_interval}_lr_{lr}_bs_{bs}_explo_{exploration[0]}_{d_t_max_time}_{nn_sizes[idx]}"
                exps_names.append(session_name)
                
                path = f"{dir_path}/{session_name}/stats"
                try :
                    if not os.path.exists(path):
                        raise(Exception("test results not found"))
                    # Get the results for the experiment
                    exp_df = convert_tb_data(path)
                    injected = exp_df[exp_df.name == "overlay_data_pkts_injected_bytes_time"].sort_values("step").iloc[-1].value
                    big_s = exp_df[exp_df.name == "overlay_big_signalling_bytes"].sort_values("step").iloc[-1].value
                    small_s = exp_df[exp_df.name == "overlay_small_signalling_bytes"].sort_values("step").iloc[-1].value
                    ping_s = exp_df[exp_df.name == "overlay_ping_signalling_bytes"].sort_values("step").iloc[-1].value
                    signalling = (big_s + small_s + ping_s)/(injected)
                    accu.append(signalling)
                    # append the results to the dataframe
                    # plot_metric_vs_sync_step(exp_df, exp_name, "test_global_cost", ax)
                    # assert len(exp_df.step.unique()) != len(test_loads) # check if all the test loads are present
                except Exception as e:
                    print(session_name)
                    not_found.append(session_name)
                    continue
                found.append(session_name)
        if len(accu) == 1:
            accu = accu * len(sync_steps)
        if len(accu) != len(sync_steps):
            raise(Exception("data not found for " + signaling_type + " " + dqn_model))
        ax.plot(sync_steps, np.array(accu), label=f"{signaling_type} {dqn_model}", marker="o")
        
plt.xlabel(xlabel="Sync step")
plt.ylabel(ylabel="Overhead ratio (signalling/injected)")
plt.tight_layout()
plt.legend()
plt.savefig(f"overhead_vs_sync_steps.png") 
# for exp_name in exps_names:
#     path = f"{dir_path}/{exp_name}/test_results"
#     try :
#         if not os.path.exists(path):
#             raise(Exception("test results not found"))
#         # Get the results for the experiment
#         exp_df = convert_tb_data(path)
#         # append the results to the dataframe
#         # plot_metric_vs_sync_step(exp_df, exp_name, "test_global_cost", ax)
#         # assert len(exp_df.step.unique()) != len(test_loads) # check if all the test loads are present
#     except Exception as e:
#         print(exp_name)
#         not_found.append(exp_name)
#         continue
#     found.append(exp_name)

# # save the figure
# print(not_found)
# print(len(not_found), len(found), len(exps_names))
"""
    exp_values = np.array(exp_df[exp_df.name == eval_metric_name].sort_values("step").value.values, dtype=float)
    
    # Skip instances that don't have test results
    if len(exp_values) == 0:
        continue
    
    # Check if the results are better than SP 
    exp_flag = int(np.all(exp_values[2:5] < sp_values[2:5]))
    
    # Compute the MSE between the exp and the opt solution
    exp_mse = np.mean(np.abs((exp_values[2:5] - opt_values[2:5])/opt_values[2:5]))
    
    # Write everything to a dict
    exp_dict = dict(
        flag=exp_flag,
        mse=exp_mse,
        exp_values=exp_values
    )
    
    # Add the parameters of the experiment
    params_names = ["ping frequency", "train load", "use throughput",  "moving average window", "learning rate", "batch size", "exploration fixed"]
    params_markers = ["freq", "load", "dqn", "avg", "lr", "bs", "explo"]
    name_splitted = exp_name.split("_")
    for i, param_marker in enumerate(params_markers):
        param_name = params_names[i]
        param_value =  name_splitted[name_splitted.index(param_marker)+1]
        if param_name == "use throughput":
            if param_value == "with":
                param_value = 1
            else:
                param_value = 0
        if param_name == "exploration fixed":
            if param_value == "fixed":
                param_value = 1
            else:
                param_value = 0
        exp_dict[param_name] = float(param_value)
        
    # Append to the dataframe
    all_exps_results.append(pd.DataFrame(exp_dict))
    
# Sort and fix the variables
all_exps_df = pd.concat(all_exps_results)[params_names+["flag", "mse"]]
all_exps_df.reset_index()
print(all_exps_df)
"""