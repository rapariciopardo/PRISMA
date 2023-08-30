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
dir_path = '/mnt/examples/5n_overlay_full_mesh_abilene/results/Compare_RCPO_offband/'
dir_path = '/home/redha/PRISMA_copy/prisma/examples/overlay_full_mesh_10n_geant/results/Compare_RCPO_offband/'

plt.rcParams.update({'font.family': "serif"})
plt.rcParams.update({'font.size': 26})
# evaluation metric name
metric_names = {"stats": ["avg_hops_over_time",
                          "signalling ratio",
                          "total_hops_over_time"], 
                "nb_new_pkts": ["pkts_over_time"],
                "nb_lost_pkts": ["pkts_over_time"],
                "nb_arrived_pkts": ["pkts_over_time"],
                f"test_results":["test_overlay_cost",
                                 "test_overlay_loss_rate", 
                                 "test_overlay_e2e_delay",
                                 "test_global_cost",
                                 "test_global_loss_rate", 
                                 "test_global_e2e_delay"]}
metric_name = "test_global_cost"
exps_names = []

#------------------------------------------------------------
# % set the parameters
# static parameters
traff_mats = [0, 2, 3, 1]
traff_mats = list(range(10))
traff_mats = [0,]
# sync_steps = list(range(1, 9))
sync_steps = list(range(1, 10))
sync_steps = [1,3]
seed = 100
rb_size = 10000
signaling_types = ["NN", "digital_twin", "sp", "opt",]
# signaling_types = ["NN",]
# dqn_models = ["", "_lite", "_lighter", "_lighter_2", "_lighter_3", "_ff"][:-1]
dqn_models = ["",]
nn_size = 35328
# variable parameters
training_loads = [0.4, 0.7, 0.9]
# training_loads = [0.9]
topology_name = "abilene"
bs = 512
lr = 0.00001
exploration = ["vary", 1.0, 0.01]
ping_freq = 10000
mv_avg_interval = 5
max_output_buffer_size = 16260
train_duration = 20
test_loads = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
loss_penalty_types = ["None", "fixed", "constrained"]
lambda_train_step = -1
buffer_soft_limit =0
snapshots = [f"episode_{i}_step_{i}" for i in range(1, 20, 1)] + ["final"]
snapshots =  ["final"]
            
#% reset the variables
all_exps_results = {}
# go through all the experiments
found = []

#------------------------------------------------------------
#% retrieve the results
not_found = []
for signaling_type in signaling_types:
    for idx, dqn_model in enumerate(dqn_models):
        for snapshot in snapshots:
            for loss_penalty_type in loss_penalty_types:
                for training_load in training_loads:
                        if signaling_type != "NN" and dqn_model != "":
                            continue
                        for traff_mat in traff_mats:
                            for sync_step in sync_steps:
                                if signaling_type != "NN":
                                    if sync_step != 1 or snapshot not in ("", "final") or loss_penalty_type != "constrained":
                                        continue 
                                if sync_step not in (1, 3) and snapshot != "final":
                                    continue
                                session_name = f"sync_{sync_step}_mat_{traff_mat}_dqn__{signaling_type}_size_{nn_size}_tr_{training_load}_sim_20_20_lr_1e-05_bs_512_outb_16260_losspen_{loss_penalty_type}_lambda_step_-1_ratio_{0}_wait_{0}_lambda_lr_1e6_dt_time_5_ping_{0.1}_vary_one_explo_first_loss_{1}_reset_{0}_use_loss_1"

                                if signaling_type in ["sp", "opt", "opt_10"]:
                                    if signaling_type == "sp":
                                        session_name = f"{signaling_type}_{traff_mat}_25_final_ping_0.1"
                                    else:
                                        if signaling_type == "opt_10":
                                            session_name = f"opt_{traff_mat}_25_final_ping_0.1_10k_buff"
                                        else:
                                            session_name = f"{signaling_type}_{traff_mat}_25_final_ping_0.1"
                                            
                                    snapshot = ""
                                if session_name in found:
                                    continue
                                exps_names.append(session_name)
                                for case in ["stats", f"test_results", "nb_new_pkts", "nb_lost_pkts", "nb_arrived_pkts"]:
                                    if case in ["stats", "nb_new_pkts", "nb_lost_pkts", "nb_arrived_pkts"] and signaling_type in ("sp", "opt"):
                                        continue
                                    if case in ["stats", "nb_new_pkts", "nb_lost_pkts", "nb_arrived_pkts"]:
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
                                            if case in ["stats", "nb_new_pkts", "nb_lost_pkts", "nb_arrived_pkts"]:
                                                data = data.iloc[-1]
                                            key = f"{signaling_type}{dqn_model}_{sync_step}_{loss_penalty_type}_{training_load}_{snapshot}_{case}_{metric_name}"
                                            if signaling_type in ["sp", "opt", "opt_10"]:
                                                key = f"{signaling_type}_{traff_mat}_{case}_{metric_name}"
                                            steps, indices = np.unique(data.step, return_index=True)
                                            if len(steps) > 1:
                                                data_to_store= np.array(data.value)[indices].reshape(-1, 1)
                                            else:
                                                data_to_store= np.array(data.value).reshape(-1, 1)
                                            if key not in all_exps_results.keys():
                                                all_exps_results[key] = {"loads": steps.tolist(),
                                                                        "traff_mat": [traff_mat,],
                                                                        "data": data_to_store}
                                            else:
                                                if traff_mat not in all_exps_results[key]["traff_mat"]:
                                                    all_exps_results[key]["data"] = np.concatenate((all_exps_results[key]["data"],
                                                                                                    data_to_store),
                                                                                                axis=1)
                                                    all_exps_results[key]["traff_mat"].append(traff_mat)
                                            if case == f"test_results":
                                                assert len(exp_df.step.unique()) == len(test_loads) # check if all the test loads are present
                                    except Exception as e:
                                        print(session_name, path)
                                        not_found.append(f'{session_name}')
                                        continue
                                    found.append(session_name)





#1)------------------------------------------------------------
#%% box plot for the metrics vs load factors over all sync steps and traffic matrices for each model
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.rcParams.update({'font.family': "Times New Roman"})
plt.rcParams.update({'font.size': 50})
plt.rcParams.update({'font.weight': 'bold'})
plt.rcParams.update({'axes.labelweight': 'bold'})
plt.rcParams.update({'axes.titleweight': 'bold'})
plt.rcParams.update({'axes.linewidth': 3})
# import seaborn as sns
for metric in metric_names[f"test_results"]:
    fig, ax = plt.subplots(figsize=(19.4, 10))
    # add dqn buffer NN model
    for snapshot in ["final"]:
        for tr_idx, training_load in enumerate(training_loads):
            bps = []
            for model_idx, loss_penalty_type in enumerate(loss_penalty_types):
                y = []
                y_min = []
                y_max = []
                x = []
                for sync_step in sync_steps:
                    name = f"NN{''}_{sync_step}_{loss_penalty_type}_{training_load}_{snapshot}_test_results"
                    temp = np.mean(all_exps_results[f"{name}_{metric}"]["data"], axis=0)
                    y.append(np.mean(temp))
                    # y_min.append(np.mean(temp) - np.std(temp))
                    # y_max.append(np.mean(temp) + np.std(temp))
                    # x.append(sync_step)
                # add box plot for each model for each training load, put the training load on the x axis and the metric on the y axis and put each model in a different color
                # plt.plot(y, label=f"{loss_penalty_type}_{training_load}".replace("None", "Loss Blind").replace("fixed", "Loss Aware").replace("constrained", "RCPO"), marker="o")
                bps.append(plt.boxplot(y, positions=[tr_idx + model_idx*0.3], widths=0.25, showfliers=False, patch_artist=True, boxprops=dict(facecolor=f"C{model_idx}"), medianprops=dict(color="black")))

                
    # add sp and opt
    # y = []
    # for mat_idx in traff_mats:
    #     y.append(np.mean(all_exps_results[f"sp_{mat_idx}_test_results_{metric}"]["data"], axis=0)[0])
    # sp_plot = plt.hlines(np.mean(y), -0.2, 2.8, label="SP", linestyle="--", color="red")
    # # plt.fill_between([-0.2, 2.8], np.min(y, axis=0), np.max(y, axis=0), alpha=0.1, color="red")
    # y = []
    # for mat_idx in traff_mats:
    #     y.append(np.mean(all_exps_results[f"opt_{mat_idx}_test_results_{metric}"]["data"], axis=0)[0])
    # opt_plot = plt.hlines(np.mean(y), -0.2, 2.8, label="OPT", linestyle="--", color="green")

    fig.tight_layout()
    metric_name = f"{''.join(metric.split('_')[1:])}".replace("cost", "Cost").replace("lossrate", "Loss Rate").replace("delay", " Delay").replace("overlay", "Overlay ").replace("global", "Global ")
    plt.xlabel(xlabel="Training Load")
    plt.ylabel(ylabel=f"{metric_name}")
    plt.xticks(ticks=[0.3, 1.3, 2.3], labels=["40%", "70%", "90%"])
    ax.legend([bp["boxes"][0] for bp in bps], ["Loss Blind", "Fixed Loss-Pen", "RCPO"], loc="upper left")
    # plt.legend(loc="upper left")
    # plt.title(f"The effect of training load in Offband setting")
    # import tikzplotlib
    # tikzplotlib.clean_figure()
    # tikzplotlib.save(f"/home/redha/PRISMA_copy/prisma/figures/new_figures/{metric_name}_vs_training_loads.tex", )
    # plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/final/{metric_name}_vs_training_loads.pdf".replace(" ", "_"),bbox_inches='tight', pad_inches=0.1)

#2)------------------------------------------------------------
#%% plot the results & save the figures for cost, loss and delay vs sync step
figures_dir = f"/home/redha/PRISMA_copy/prisma/figures"
costs = {}
for metric in metric_names[f"test_results"]: 
    fig, ax = plt.subplots(figsize=(19.4, 10))
    # add dqn buffer NN model
    for training_load in [0.9]:
        for snapshot in ["final"]:
            for loss_penalty_type in loss_penalty_types:
                for model in dqn_models:
                    y = []
                    y_min = []
                    y_max = []
                    x = []
                    for sync_step in sync_steps:
                        name = f"NN{model}_{sync_step}_{loss_penalty_type}_{training_load}_{snapshot}_test_results"
                        temp = np.mean(all_exps_results[f"{name}_{metric}"]["data"], axis=0)
                        y.append(np.mean(temp))
                        y_min.append(np.mean(temp) - np.std(temp))
                        y_max.append(np.mean(temp) + np.std(temp))
                        x.append(sync_step)
                    # add the plot into the figure with min and max values shaded area
                    if metric == "test_global_cost" and loss_penalty_type == "constrained"  :
                        costs[f"NN"] = y
                    ax.plot(x, y, 
                            label=f"{loss_penalty_type}".replace("None", "Loss Blind").replace("fixed", "Loss Aware").replace("constrained", "RCPO").replace("digital twin", "Digital Twin"),
                                linestyle="solid",
                             marker="o",
                             linewidth=7,
                             markersize=20,)
                    # ax.plot(x, y_min, 
                    #         label=f"{loss_penalty_type}_{training_load}".replace("None", "Loss Blind").replace("fixed", "Loss Aware").replace("constrained", "RCPO"),
                    #             linestyle="dashed",
                    #          marker="o",
                    #          linewidth=7,
                    #          markersize=20) 
                    # ax.plot(x, y_max, 
                    #         label=f"{loss_penalty_type}_{training_load}".replace("None", "Loss Blind").replace("fixed", "Loss Aware").replace("constrained", "RCPO"),
                    #         linestyle="dashed",
                    #          marker="o",
                    #          linewidth=7,
                    #          markersize=20)   
                    # add digital twin
    # temp = np.mean(all_exps_results[f"digital_twin_1_1_{metric}"]["data"], axis=0)
    # ax.hlines(np.mean(temp), 1, 8, label="DT 1", linestyle="--", color="blue")
    # add digital twin
    temp = np.mean(all_exps_results[f"digital_twin_3_constrained_0.9_final_test_results_{metric}"]["data"], axis=0)
    # ax.hlines(np.mean(temp), 1, 9.5, label="digital twin", linestyle="--", linewidth=7)
    costs[f"digital_twin"] = np.mean(temp)
    # # # add digital twin
    # # temp = np.mean(all_exps_results[f"digital_twin_1_10_{metric}"]["data"], axis=0)
    # # ax.hlines(np.mean(temp), 1, 8, label="DT 10", linestyle="--")
    # # add target
    temp = np.mean(all_exps_results[f"target_3_constrained_0.9_final_test_results_{metric}"]["data"], axis=0)
    costs[f"target"] = np.mean(temp)
    # ax.hlines(np.mean(temp), 1, 8, label=f"target", linestyle="--", color="grey", linewidth=7)
    
    # add sp and opt
    y = []
    for tr_idx in traff_mats:
        y.append(np.mean(all_exps_results[f"sp_{tr_idx}_test_results_{metric}"]["data"], axis=0)[0])
    ax.hlines(np.mean(y), min(sync_steps), max(sync_steps), label="OSPF", linestyle="--", color="red", 
                             linewidth=7)
    # ax.fill_between(sync_steps, np.min(y, axis=0), np.max(y, axis=0), alpha=0.2, color="red")
    y = []
    for tr_idx in traff_mats:
        y.append(np.mean(all_exps_results[f"opt_{tr_idx}_test_results_{metric}"]["data"], axis=0)[0])
    ax.hlines(np.mean(y), min(sync_steps), max(sync_steps), label="Oracle Routing", linestyle="--", color="tab:purple", linewidth=7)

    plt.xlabel(xlabel="Sync step")
    metric_name = f"{''.join(metric.split('_')[1:])}".replace("cost", "Cost").replace("lossrate", "Loss Rate").replace("delay", " Delay").replace("overlay", "Overlay ").replace("global", "Global ")    
    plt.ylabel(ylabel=f"{metric_name}")
    plt.tight_layout()
    plt.legend(loc="upper left")
    # plt.title(f"The effect of sync step in Offband setting for 90% training load")
    
    # plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/final/{metric_name}_vs_sync_steps.pdf".replace(" ", "_"),bbox_inches='tight', pad_inches=0.1)
    
#3)------------------------------------------------------------
#%% plot the results & save the figures for cost, loss and delay vs load factors 
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.rcParams.update({'font.family': "Times New Roman"})
plt.rcParams.update({'font.size': 50})
plt.rcParams.update({'font.weight': 'bold'})
plt.rcParams.update({'axes.labelweight': 'bold'})
plt.rcParams.update({'axes.titleweight': 'bold'})
plt.rcParams.update({'axes.linewidth': 3})
figures_dir = f"/home/redha/PRISMA_copy/prisma/figures"
for sync_step in [1]:
    # add dqn buffer NN model
    for metric in metric_names[f"test_results"]: 
        fig, ax = plt.subplots(figsize=(19.4, 10))
        for training_load in [0.9]:
            for snapshot in ["final"]:
                for signaling_type in ["NN",]:
                    for loss_penalty_type in loss_penalty_types:
                            y = []
                            y_min = []
                            y_max = []
                            x = []
                            if loss_penalty_type == "constrained":
                                sync_step = 1
                            elif loss_penalty_type == "fixed":
                                sync_step = 3
                            elif loss_penalty_type == "None":
                                sync_step = 3
                            name = f"{signaling_type}{''}_{sync_step}_{loss_penalty_type}_{training_load}_{snapshot}_test_results"
                            # temp = np.mean(all_exps_results[f"{name}_{metric}"]["data"], axis=1)
                            temp = all_exps_results[f"{name}_{metric}"]["data"][:, 0]
                            y.append(np.mean(temp))
                            y_min.append(np.mean(temp) - np.std(temp))
                            y_max.append(np.mean(temp) + np.std(temp))
                            x.append(sync_step)
                            # add the plot into the figure with min and max values shaded area
                            ax.plot(all_exps_results[f"{name}_{metric}"]["loads"], temp,
                                    label=f"{loss_penalty_type}".replace("None", "Loss Blind").replace("fixed", "Loss Aware").replace("constrained", "RCPO"), linestyle="solid",
                             marker="o",
                             linewidth=7,
                             markersize=20)
                            # ax.fill_between(x, y_min, y_max, alpha=0.2)
        # add digital twin
        # temp = np.mean(all_exps_results[f"digital_twin_1_1_{metric}"]["data"], axis=0)
        # ax.hlines(np.mean(temp), 1, 8, label="DT 1", linestyle="--", color="blue")
        # add digital twin
        # temp = np.mean(all_exps_results[f"digital_twin_1_5_{metric}"]["data"], axis=0)
        # ax.hlines(np.mean(temp), 1, 8, label="DT", linestyle="--")
        # # add digital twin
        # temp = np.mean(all_exps_results[f"digital_twin_1_10_{metric}"]["data"], axis=0)
        # ax.hlines(np.mean(temp), 1, 8, label="DT 10", linestyle="--")
        # add target
        # temp = np.mean(all_exps_results[f"target_1_5_{metric}"]["data"], axis=0)
        # ax.hlines(np.mean(temp), 1, 8, label=f"target", linestyle="--", color="grey")
        
        # add sp and opt
        for name in ["sp", "opt"]:
            y = []
            for tr_idx in traff_mats[:1]:
                y.append(all_exps_results[f"{name}_{tr_idx}_test_results_{metric}"]["data"][:, 0])
            ax.plot(all_exps_results[f"{name}_0_test_results_{metric}"]["loads"], np.mean(y, axis=0), label=f"{name}".replace("sp", "OSPF").replace("Oracle Routing", "Oracle Routing"), linestyle="--", marker="o", linewidth=7, markersize=20, )
        # y = all_exps_results[f"opt_0_{metric}"]["data"]
        # ax.plot(all_exps_results[f"sp_0_{metric}"]["loads"], y, label="OPT", linestyle="--", color="green")
        metric_name = f"{''.join(metric.split('_')[1:])}".replace("cost", "Cost").replace("lossrate", "Loss Rate").replace("delay", " Delay").replace("overlay", "Overlay ").replace("global", "Global ")

        plt.xlabel(xlabel="Load Factor")
        plt.xticks(ticks=[60, 70, 80, 90, 100, 110, 120], labels=["60%", "70%", "80%", "90%", "100%", "110%", "120%"])
        plt.ylabel(ylabel=f"{metric_name}")
        plt.tight_layout()
        plt.legend(loc="upper left")
        # plt.title(f"Variation of {metric_name} with load factor in Offband setting")
        # plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/final/{metric_name}_vs_loads.pdf".replace(" ", "_"),bbox_inches='tight', pad_inches=0.1)
    


#4)------------------------------------------------------------
#%% plot the results & save the figures for overhead vs sync step
overheads =  {"NN": [],
              "digital_twin": [],
              "target": [],}
fig, ax = plt.subplots(figsize=(19.4, 10))
# add dqn buffer NN model
for signaling_type in signaling_types[:-2]:
    for loss_penalty_type in ["constrained"]:
        for snapshot in snapshots:
            for training_load in [0.9]:
                y = []
                y_min = []
                y_max = []
                x = []
                for sync_step in sync_steps:
                    if signaling_type != "NN":
                        temp = all_exps_results[f"{signaling_type}_{3}_{loss_penalty_type}_{training_load}_{snapshot}_stats_signalling ratio"]['data'].max()
                    else:
                        temp = all_exps_results[f"{signaling_type}_{3}_{loss_penalty_type}_{training_load}_{snapshot}_stats_signalling ratio"]['data'].max() + ((69*4*5*400/sync_step)/(all_exps_results[f"{signaling_type}_{sync_step}_{loss_penalty_type}_{training_load}_{snapshot}_nb_new_pkts_pkts_over_time"]['data'].max()*20))
                    y.append(np.mean(temp))
                    y_min.append(np.mean(temp) - np.std(temp))
                    y_max.append(np.mean(temp) + np.std(temp))
                    x.append(sync_step)
                # add the plot into the figure with min and max values shaded area
                overheads[signaling_type] = y
                ax.plot(x, y, label=f"{signaling_type}".replace("NN", "Model Sharing").replace("digital_twin", "Digital Twin").replace("target", "Target Value Sharing").replace("sp", "OSPF").replace("opt", "Oracle Routing"), marker="o",
                             linewidth=7, markersize=20)
                ax.fill_between(x, y_min, y_max, alpha=0.2)
# add digital twin
# temp = (all_exps_results["digital_twin" + "" + f"_{1}_5_overlay_big_signalling_bytes"]["data"] + all_exps_results["digital_twin" + "" + f"_{1}_5_overlay_small_signalling_bytes"]["data"])/all_exps_results["NN" + "" + f"_{1}_5_overlay_data_pkts_injected_bytes_time"]["data"]
# plt.hlines(np.mean(temp), 1, 8, label="DT", linestyle="--")
#add target
# temp = (all_exps_results["target" + "" + f"_{1}_5_overlay_big_signalling_bytes"]["data"] + all_exps_results["target" + "" + f"_{1}_5_overlay_small_signalling_bytes"]["data"])/all_exps_results["NN" + "" + f"_{1}_5_overlay_data_pkts_injected_bytes_time"]["data"]
# plt.hlines(np.mean(temp), 1, 8, label=f"target", linestyle="--", color="grey")
plt.xlabel(xlabel="Sync step")
plt.ylabel(ylabel=f"Overhead Ratio")
plt.yticks(ticks=[0.4, 2.5, 5], labels=["40%", "250%", "500%"])
plt.legend()
plt.tight_layout()
plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/final/overhead_ratio_vs_sync_steps.pdf",bbox_inches='tight', pad_inches=0.1)




#5)------------------------------------------------------------

#%% plot the results & save the figures for cost vs overhead ratio
fig, ax = plt.subplots(figsize=(19.4, 10))
# add dqn buffer NN model
ax.plot(overheads["NN"], costs["NN"], label=f"NN".replace("NN", "Model Sharing"), marker="o", linewidth=7, markersize=20)
# add digital twin
ax.plot(overheads["digital_twin"][0], costs["digital_twin"], label=f"DT".replace("digital_twin", "Digital Twin"), marker="*", color="black", markersize=30, linewidth=0,)
# add target
ax.plot(overheads["target"][0], costs["target"], label=f"target".replace("target", "Target Value Sharing"), marker="*", color="grey", markersize=30, linewidth=0,)
plt.xlabel(xlabel="Overhead Ratio")
plt.ylabel(ylabel=f"Average Cost")
plt.tight_layout()
plt.legend()
plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/final/overhead_ratio_vs_cost.pdf",bbox_inches='tight', pad_inches=0.1)





#%%
for model in dqn_models:
    y = []
    y_min = []
    y_max = []
    x = []
    for sync_step in sync_steps:
        temp1 = np.mean(all_exps_results["NN" + model + f"_{sync_step}_constrained_0.9_final_test_results_test_global_cost"]["data"], axis=0)
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
#5)------------------------------------------------------------
#%% plot the results & save the figures for overhead vs sync step
fig, ax = plt.subplots(figsize=(19.4, 10))
# add dqn buffer NN model
for signaling_type in signaling_types[:-2]:
    for loss_penalty_type in ["constrained"]:
        for snapshot in snapshots:
            for training_load in [0.9]:
                y = []
                y_min = []
                y_max = []
                x = []
                for sync_step in sync_steps:
                    if signaling_type != "NN":
                        temp = all_exps_results[f"{signaling_type}_{3}_{loss_penalty_type}_{training_load}_{snapshot}_stats_signalling ratio"]['data'].max()
                    else:
                        temp = all_exps_results[f"{signaling_type}_{sync_step}_{loss_penalty_type}_{training_load}_{snapshot}_test_results_test_global_cost"]['data']
                    y.append(np.mean(temp))
                    y_min.append(np.mean(temp) - np.std(temp))
                    y_max.append(np.mean(temp) + np.std(temp))
                    x.append(sync_step)
                # add the plot into the figure with min and max values shaded area
                overheads[signaling_type] = y
                ax.plot(x, y, label=f"{signaling_type}", marker="o",
                             linewidth=7, markersize=20)
                ax.fill_between(x, y_min, y_max, alpha=0.2)
# add digital twin
# temp = (all_exps_results["digital_twin" + "" + f"_{1}_5_overlay_big_signalling_bytes"]["data"] + all_exps_results["digital_twin" + "" + f"_{1}_5_overlay_small_signalling_bytes"]["data"])/all_exps_results["NN" + "" + f"_{1}_5_overlay_data_pkts_injected_bytes_time"]["data"]
# plt.hlines(np.mean(temp), 1, 8, label="DT", linestyle="--")
#add target
# temp = (all_exps_results["target" + "" + f"_{1}_5_overlay_big_signalling_bytes"]["data"] + all_exps_results["target" + "" + f"_{1}_5_overlay_small_signalling_bytes"]["data"])/all_exps_results["NN" + "" + f"_{1}_5_overlay_data_pkts_injected_bytes_time"]["data"]
# plt.hlines(np.mean(temp), 1, 8, label=f"target", linestyle="--", color="grey")
plt.xlabel(xlabel="Sync step")
plt.ylabel(ylabel=f"Overhead Ratio")
plt.tight_layout()
plt.legend()
# plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/final/overhead_ratio_vs_sync_steps.jpeg",bbox_inches='tight', pad_inches=0.1)
# plt.savefig(f"{figures_dir}/overhead_ratio_vs_sync_steps.png") 




#%% plot the results & save the figures for average cost, loss and delay over load factors vs episodes for each sync step 

figures_dir = f"/home/redha/PRISMA_copy/prisma/figures"
sync_step = 1
# snapshots.remove("episode_13_step_13")
for metric in metric_names[f"test_results"]: 
    fig, ax = plt.subplots(figsize=(19.4, 10))
    # add dqn buffer NN model
    for training_load in [0.9, ]:
            for loss_penalty_type in loss_penalty_types:
                for model in dqn_models:
                    y = []
                    y_min = []
                    y_max = []
                    x = []
                    for idx, snapshot in enumerate(snapshots):
                        name = f"NN{model}_{sync_step}_{loss_penalty_type}_{training_load}_{snapshot}_test_results_test_results"
                        temp = np.mean(all_exps_results[f"{name}_{metric}"]["data"], axis=0)
                        y.append(np.mean(temp))
                        y_min.append(np.mean(temp) - np.std(temp))
                        y_max.append(np.mean(temp) + np.std(temp))
                        x.append(idx)
                    # add the plot into the figure with min and max values shaded area
                    ax.plot(x, y, 
                            label=f"{loss_penalty_type}_{training_load}".replace("None", "Loss Blind").replace("fixed", "Loss Aware").replace("constrained", "RCPO"),
                            marker=".")
                    # ax.fill_between(x, y_min, y_max, alpha=0.2)
    # add digital twin
    # temp = np.mean(all_exps_results[f"digital_twin_1_1_{metric}"]["data"], axis=0)
    # ax.hlines(np.mean(temp), 1, 8, label="DT 1", linestyle="--", color="blue")
    # add digital twin
    # temp = np.mean(all_exps_results[f"digital_twin_1_5_{metric}"]["data"], axis=0)
    # ax.hlines(np.mean(temp), 1, 8, label="DT", linestyle="--")
    # # add digital twin
    # temp = np.mean(all_exps_results[f"digital_twin_1_10_{metric}"]["data"], axis=0)
    # ax.hlines(np.mean(temp), 1, 8, label="DT 10", linestyle="--")
    # add target
    # temp = np.mean(all_exps_results[f"target_1_5_{metric}"]["data"], axis=0)
    # ax.hlines(np.mean(temp), 1, 8, label=f"target", linestyle="--", color="grey")
    
    # add sp and opt
    y = np.mean(all_exps_results[f"sp_0_{metric}"]["data"], axis=0)
    # ax.hlines(np.mean(y), 1, 5, label="SP", linestyle="--", color="red")
    # y = np.mean(all_exps_results[f"opt_0_{metric}"]["data"], axis=0)
    # ax.hlines(np.mean(y), 1, 5, label="OPT", linestyle="--", color="green")

    plt.xlabel(xlabel="Episode")
    plt.ylabel(ylabel=f"{''.join(metric.split('_')[-2:])}")
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.title(f"{metric} over loads for sync step {sync_step}")
    # plt.savefig(f"{figures_dir}/{''.join(metric.split('_')[-2:])}_vs_sync_steps_new.png") 
























 
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
