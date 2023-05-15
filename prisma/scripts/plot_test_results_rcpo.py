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
#------------------------------------------------------------
#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append('/home/redha/PRISMA_copy/prisma/source/')
from utils import convert_tb_data

def plot_metric_vs_sync_step(result_df, exps_name, metric_name, axis):
    # plot the metric vs. sync step given a dataframes of results
    data = result_df[result_df.name == metric_name].sort_values("step")
    x = data.step.values
    y = data.value.values
    axis.plot(x, y, label=exps_name)

    

# The directory where the test results are stored
dir_path = '/home/redha/PRISMA_copy/prisma/examples/abilene/results/ITC_NN_size_variations_experiment_rcpo_fixed'

plt.rcParams.update({'font.family': "serif"})
plt.rcParams.update({'font.size': 26})
# evaluation metric name
metric_names = {"stats": ["avg_hops_over_time", "overlay_big_signalling_bytes", "overlay_small_signalling_bytes", "overlay_data_pkts_injected_bytes_time", "overlay_ping_signalling_bytes"], 
                f"test_results":["test_global_cost", "test_global_loss_rate", "test_global_e2e_delay"]}
metric_name = "test_global_cost"
exps_names = []

#------------------------------------------------------------
#%% set the parameters
# static parameters
# traff_mats = [0, 2, 3, 1]
traff_mats = [0,1, 2, 3]
# traff_mats = [2,]
# sync_steps = list(range(1, 9))
sync_steps = [1,]
seed = 100
rb_size = 10000
# signaling_types = ["NN", "digital-twin", "target"]
signaling_types = ["NN", "sp", "opt"]
# dqn_models = ["", "_lite", "_lighter", "_lighter_2", "_lighter_3", "_ff"][:-1]
dqn_models = ["",]
nn_sizes = [35328,]
# variable parameters
training_load = 0.4
topology_name = "abilene"
bs = 1024
lr = 0.0001
exploration = ["vary", 1.0, 0.01]
ping_freq = 10000
mv_avg_interval = 100
max_output_buffer_size = 16260
train_duration = 60
test_loads = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
loss_penalty_types = ["constrained", "fixed"]
lambda_train_step = 1
buffer_soft_limits = [0.1, 0.4, 0.6]
snapshots = ["episode_1_step_1",
            "episode_1_step_2",
            "episode_1_step_3",
            "episode_1_step_4",
            "episode_1_step_5",
            "final"]
            
lambda_waits = [0, 25, 60]
#%% reset the variables
all_exps_results = {}
# go through all the experiments
found = []

#------------------------------------------------------------
#%% retrieve the results
not_found = []
for signaling_type in signaling_types:
    for idx, dqn_model in enumerate(dqn_models):
        for snapshot in snapshots:
            for loss_penalty_type in loss_penalty_types:
                for buffer_soft_limit in buffer_soft_limits:
                    for lambda_wait in lambda_waits:
                        if signaling_type != "NN" and dqn_model != "":
                            continue
                        if loss_penalty_type != "constrained":
                                if not(lambda_wait == 0 and buffer_soft_limit == 0.6):
                                    continue
                        else:
                            if lambda_wait == 60 and (buffer_soft_limit!=0.1):
                                continue
                        for traff_mat in traff_mats:
                            for sync_step in sync_steps:
                                if signaling_type != "NN":
                                    if sync_step != 1 or snapshot != "final" or lambda_wait != 0 or buffer_soft_limit != 0.6 or loss_penalty_type != "fixed":
                                        continue 
                                session_name = f"sync_{sync_step}_traff_mat_{traff_mat}_dqn_buffer{dqn_model}_{signaling_type}_rb_size_{rb_size}_{topology_name}_train_load_{training_load}_simTime_{train_duration}_lr_{lr}_bs_{bs}_explo_{exploration[0]}_natural_output_buffer_{max_output_buffer_size}_loss_pen_{loss_penalty_type}_lambda_step_{lambda_train_step}_soft_limit_ratio_{buffer_soft_limit}_wait_{lambda_wait}_lambda_lr_1e5"

                                if signaling_type in ["sp", "opt"]:
                                    session_name = f"{signaling_type}_{traff_mat}"
                                    snapshot = "None"
                                if session_name in found:
                                    continue
                                exps_names.append(session_name)
                                for case in ["stats", f"test_results"]:
                                    if case == "stats" and signaling_type != "NN":
                                        continue
                                    if case == "stats":
                                        path = f"{dir_path}/{session_name}/{case}"
                                    else:
                                        path = f"{dir_path}/{session_name}/{case}/{snapshot}"
                                    try :
                                        if not os.path.exists(path):
                                            raise(Exception(f"{case} results not found"))
                                        # Get the results for the experiment
                                        exp_df = convert_tb_data(path)
                                        for metric_name in metric_names[case]:
                                            data = exp_df[exp_df.name == metric_name].sort_values("step")
                                            if case == "stats":
                                                data = data.iloc[-1]
                                            key = f"{signaling_type}{dqn_model}_{sync_step}_{loss_penalty_type}_{lambda_wait}_{buffer_soft_limit}_{snapshot}_{traff_mat}_{metric_name}"
                                            if signaling_type != "NN":
                                                key = f"{signaling_type}_{traff_mat}_{metric_name}"
                                            if key not in all_exps_results.keys():
                                                all_exps_results[key] = {"loads": np.array(data.step).tolist(),
                                                                        "traff_mat": [traff_mat,],
                                                                        "data": np.array(data.value).reshape(-1, 1)}
                                            else:
                                                all_exps_results[key]["data"] = np.concatenate((all_exps_results[key]["data"],
                                                                                                np.array(data.value).reshape(-1, 1)),
                                                                                            axis=1)
                                                all_exps_results[key]["traff_mat"].append(traff_mat)
                                            if case == f"test_results":
                                                assert len(exp_df.step.unique()) == len(test_loads) # check if all the test loads are present
                                    except Exception as e:
                                        print(session_name, key)
                                        not_found.append((session_name, key))
                                        continue
                                    found.append((session_name, key))

#------------------------------------------------------------
#%% plot the results & save the figures for cost, loss and delay vs sync step
figures_dir = f"/home/redha/PRISMA_copy/prisma/figures"
for metric in metric_names[f"test_results"]: 
    fig, ax = plt.subplots(figsize=(19.4, 10))
    # add dqn buffer NN model
    for model in dqn_models:
        y = []
        y_min = []
        y_max = []
        x = []
        for sync_step in sync_steps:
            temp = np.mean(all_exps_results["NN" + model + f"_{sync_step}_5_{metric}"]["data"], axis=0)
            y.append(np.mean(temp))
            y_min.append(np.mean(temp) - np.std(temp))
            y_max.append(np.mean(temp) + np.std(temp))
            x.append(sync_step)
        # add the plot into the figure with min and max values shaded area
        ax.plot(x, y, label=f"NN{model}", marker="o")
        # ax.fill_between(x, y_min, y_max, alpha=0.2)
    # add digital twin
    # temp = np.mean(all_exps_results[f"digital_twin_1_1_{metric}"]["data"], axis=0)
    # ax.hlines(np.mean(temp), 1, 8, label="DT 1", linestyle="--", color="blue")
    # add digital twin
    temp = np.mean(all_exps_results[f"digital_twin_1_5_{metric}"]["data"], axis=0)
    ax.hlines(np.mean(temp), 1, 8, label="DT", linestyle="--")
    # # add digital twin
    # temp = np.mean(all_exps_results[f"digital_twin_1_10_{metric}"]["data"], axis=0)
    # ax.hlines(np.mean(temp), 1, 8, label="DT 10", linestyle="--")
    # add target
    temp = np.mean(all_exps_results[f"target_1_5_{metric}"]["data"], axis=0)
    ax.hlines(np.mean(temp), 1, 8, label=f"target", linestyle="--", color="grey")
    
    # add sp and opt
    y = np.mean(all_exps_results[f"sp_1_5_{metric}"]["data"], axis=0)
    ax.hlines(np.mean(y), 1, 8, label="SP", linestyle="--", color="red")
    y = np.mean(all_exps_results[f"opt_1_5_{metric}"]["data"], axis=0)
    ax.hlines(np.mean(y), 1, 8, label="OPT", linestyle="--", color="green")

    plt.xlabel(xlabel="Sync step")
    plt.ylabel(ylabel=f"{''.join(metric.split('_')[-2:])}")
    plt.tight_layout()
    plt.legend(loc="upper left")
    # plt.savefig(f"{figures_dir}/{''.join(metric.split('_')[-2:])}_vs_sync_steps_new.png") 
 
#%% plot the results & save the figures for overhead vs sync step
fig, ax = plt.subplots(figsize=(19.4, 10))
# add dqn buffer NN model
for model in dqn_models:
    y = []
    y_min = []
    y_max = []
    x = []
    for sync_step in sync_steps:
        temp = (all_exps_results["NN" + model + f"_{sync_step}_5_overlay_big_signalling_bytes"]["data"] + all_exps_results["NN" + model + f"_{sync_step}_5_overlay_small_signalling_bytes"]["data"])/all_exps_results["NN" + model + f"_{sync_step}_5_overlay_data_pkts_injected_bytes_time"]["data"]
        y.append(np.mean(temp))
        y_min.append(np.mean(temp) - np.std(temp))
        y_max.append(np.mean(temp) + np.std(temp))
        x.append(sync_step)
    # add the plot into the figure with min and max values shaded area
    ax.plot(x, y, label=f"NN{model}", marker="o")
    ax.fill_between(x, y_min, y_max, alpha=0.2)
# add digital twin
temp = (all_exps_results["digital_twin" + "" + f"_{1}_5_overlay_big_signalling_bytes"]["data"] + all_exps_results["digital_twin" + "" + f"_{1}_5_overlay_small_signalling_bytes"]["data"])/all_exps_results["NN" + "" + f"_{1}_5_overlay_data_pkts_injected_bytes_time"]["data"]
plt.hlines(np.mean(temp), 1, 8, label="DT", linestyle="--")
#add target
temp = (all_exps_results["target" + "" + f"_{1}_5_overlay_big_signalling_bytes"]["data"] + all_exps_results["target" + "" + f"_{1}_5_overlay_small_signalling_bytes"]["data"])/all_exps_results["NN" + "" + f"_{1}_5_overlay_data_pkts_injected_bytes_time"]["data"]
plt.hlines(np.mean(temp), 1, 8, label=f"target", linestyle="--", color="grey")
plt.xlabel(xlabel="Sync step")
plt.ylabel(ylabel=f"Overhead Ratio")
plt.tight_layout()
plt.legend()
# plt.savefig(f"{figures_dir}/overhead_ratio_vs_sync_steps.png") 

#------------------------------------------------------------

#%% plot the results & save the figures for cost vs overhead ratio
fig, ax = plt.subplots(figsize=(19.4, 10))
for model in dqn_models:
    y = []
    y_min = []
    y_max = []
    x = []
    for sync_step in sync_steps:
        temp1 = np.mean(all_exps_results["NN" + model + f"_{sync_step}_5_test_global_cost"]["data"], axis=0)
        temp2 = (all_exps_results["NN" + model + f"_{sync_step}_5_overlay_big_signalling_bytes"]["data"] + all_exps_results["NN" + model + f"_{sync_step}_5_overlay_small_signalling_bytes"]["data"])/all_exps_results["NN" + model + f"_{sync_step}_5_overlay_data_pkts_injected_bytes_time"]["data"]
        y.append(np.mean(temp1))
        x.append(np.mean(temp2))
        y_min.append(np.mean(temp1) - np.std(temp1))
        y_max.append(np.mean(temp1) + np.std(temp1))
    # add the plot into the figure with min and max values shaded area
    ax.plot(x, y, label=f"NN{model}", marker="o")
    ax.fill_between(x, y_min, y_max, alpha=0.2)

# add digital twin
temp1 = np.mean(all_exps_results["digital_twin" + "" + f"_{1}_5_test_global_cost"]["data"], axis=0)
temp2 = (all_exps_results["digital_twin" + "" + f"_{1}_5_overlay_big_signalling_bytes"]["data"] + all_exps_results["digital_twin" + "" + f"_{1}_5_overlay_small_signalling_bytes"]["data"])/all_exps_results["NN" + "" + f"_{1}_5_overlay_data_pkts_injected_bytes_time"]["data"]
ax.plot(np.mean(temp2), np.mean(temp1), label=f"DT", marker="*", color="black", markersize=10)

# add target
temp1 = np.mean(all_exps_results["target" + "" + f"_{1}_5_test_global_cost"]["data"], axis=0)
temp2 = (all_exps_results["target" + "" + f"_{1}_5_overlay_big_signalling_bytes"]["data"] + all_exps_results["target" + "" + f"_{1}_5_overlay_small_signalling_bytes"]["data"])/all_exps_results["NN" + "" + f"_{1}_5_overlay_data_pkts_injected_bytes_time"]["data"]
ax.plot(np.mean(temp2), np.mean(temp1), label=f"target", marker="*", color="grey", markersize=10)

# add sp and opt
y = np.mean(all_exps_results[f"sp_1_5_test_global_cost"]["data"], axis=0)
ax.hlines(np.mean(y), 0, 6, label="SP", linestyle="--", color="red")
y = np.mean(all_exps_results[f"opt_1_5_test_global_cost"]["data"], axis=0)
ax.hlines(np.mean(y), 0, 6, label="OPT", linestyle="--", color="green")
plt.ylabel(ylabel="Average Cost")
plt.xlabel(xlabel=f"Overhead Ratio")
plt.tight_layout()
plt.legend()
# plt.savefig(f"{figures_dir}/overhead_ratio_vs_cost.png") 
 
#%% plot the results & save the figures for cost, loss and delay vs load factors
# key = f"{signaling_type}{dqn_model}_{sync_step}_{loss_pen_type}_{lambda_wait}_{buffer_soft_limit}_{traff_mat}_{metric_name}"
model_key = [
            "NN_1_constrained_0_0.1_final",
            "NN_1_constrained_0_0.4_final",
            "NN_1_constrained_0_0.6_final",            
            "NN_1_constrained_25_0.1_final",
            "NN_1_constrained_25_0.4_final",
            "NN_1_constrained_25_0.6_final",
            "NN_1_constrained_60_0.1_final",
            "NN_1_fixed_0_0.6_final",
            ]
model_names = [
               "RCPO after 0s 10%",
               "RCPO after 0s 40%",
               "RCPO after 0s 60%",
                "RCPO after 25s 10%",
               "RCPO after 25s 40%",
               "RCPO after 25s 60%",
               "loss blind",
               "Loss penalty",
               ]
mat = 0
for metric in metric_names[f"test_results"][:1]: 
    # add dqn buffer NN model
    fig, ax = plt.subplots(figsize=(19.4, 10))
    for i in range(len(model_names)):
        y = []
        temp = []
        y_min = []
        y_max = []
        x = []
        # for sync_step in sync_steps:
        for mat in [0,]:
            if len(temp) == 0:
                temp = all_exps_results[model_key[i] + f"_{mat}_{metric}"]["data"][:]
                
            else:
                temp = np.concatenate((temp, all_exps_results[model_key[i] + f"_{mat}_{metric}"]["data"]), axis=1)
        y = np.mean(temp, axis=1)
        print(f"{model_names[i]}", np.mean(y))
        y_min = np.min(temp, axis=1)
        y_max = np.max(temp, axis=1)
        x = all_exps_results[model_key[i] + f"_{mat}_{metric}"]["loads"]
        # add the plot into the figure with min and max values shaded area
        ax.plot(x, y, label=f"{model_names[i]}", marker="o")
        # ax.fill_between(x, y_min, y_max, alpha=0.2)
            

    plt.xlabel(xlabel="Load Factor")
    plt.ylabel(ylabel=f"{''.join(metric.split('_')[-2:])}")
    plt.tight_layout()
    plt.legend()
    # plt.savefig(f"{figures_dir}/{''.join(metric.split('_')[-2:])}_vs_sync_steps_new.png") 
 
#%% plot the results for the average cost vs epochs 
model_key = [
            # "NN_1_constrained_0_0.1_",
            "NN_1_constrained_0_0.4_",
            # "NN_1_constrained_0_0.6_",            
            # "NN_1_constrained_25_0.1_",
            "NN_1_constrained_25_0.4_",
            # "NN_1_constrained_25_0.6_",
            "NN_1_constrained_60_0.1_",
            "NN_1_fixed_0_0.6_",
            "sp",
            "opt"
            ]
model_names = [
            #    "RCPO after 0s 10%",
               "RCPO after 0s 40%",
            #    "RCPO after 0s 60%",
            #     "RCPO after 25s 10%",
               "RCPO after 25s 40%",
            #    "RCPO after 25s 60%",
               "loss blind",
               "Loss penalty",
            "SP", 
            "OPT"
               ]
snapshots = snapshots[:-1]
fig, ax = plt.subplots(figsize=(19.4, 10))
for i in range(len(model_key)):
    y = []
    for snapshot in snapshots:
        if model_key[i] in ("sp", "opt"):
                snapshot = ""
        for metric in metric_names[f"test_results"][:1]: 
            temp = []
            y_min = []
            y_max = []
            x = []
            # for sync_step in sync_steps:
            for mat in [0,]:
                if len(temp) == 0:
                    temp = all_exps_results[model_key[i] + f"{snapshot}_{mat}_{metric}"]["data"][:]
                    
                else:
                    temp = np.concatenate((temp, all_exps_results[model_key[i] + f"{snapshot}_{mat}_{metric}"]["data"]), axis=1)
            # y = np.mean(temp, axis=1)
            print(f"{model_names[i]}", np.mean(temp))
            y.append(np.mean(temp))
            y_min = np.min(temp, axis=1)
            y_max = np.max(temp, axis=1)
            x = all_exps_results[model_key[i] + f"{snapshot}_{mat}_{metric}"]["loads"]
            # add the plot into the figure with min and max values shaded area
    ax.plot(range(len(snapshots)), y, label=f"{model_names[i]}", marker="o")
    # ax.fill_between(x, y_min, y_max, alpha=0.2)
        

    plt.xlabel(xlabel="Snapshot Number")
    plt.xticks(range(len(snapshots)), [" ".join(val.split("_")[-1:]) for val in snapshots])
    plt.ylabel(ylabel=f"{''.join(metric.split('_')[-2:])}")
    plt.tight_layout()
    # plt.title(f"{model_names[i]}")
    plt.legend()
# %%
