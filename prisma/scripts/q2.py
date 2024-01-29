
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
import pandas as pd
sys.path.append('/home/redha/PRISMA_copy/prisma/source/')
from utils import convert_tb_data

def plot_metric_vs_sync_step(result_df, exps_name, metric_name, axis):
    # plot the metric vs. sync step given a dataframes of results
    data = result_df[result_df.name == metric_name].sort_values("step")
    x = data.step.values
    y = data.value.values
    axis.plot(x, y, label=exps_name)

    

# The directory where the test results are stored
dir_path = '/home/redha/PRISMA_copy/prisma/examples/5n_overlay_full_mesh_abilene/results/Final_results'
# dir_path = '/mnt/journal_paper_results/ex/5n_overlay_full_mesh_abilene/results/Final_results'
dir_path = '/mnt/backup_examples_new/5n_overlay_full_mesh_abilene/results/abilene_results_with_threshold'
dir_path = '/mnt/backup_examples_new/overlay_full_mesh_10n_geant/results/geant_results_with_threshold'
# dir_path = '/mnt/final_results_geant'

plot_style = {'font.family': "Times New Roman",
                     'font.size': 50,
                     'font.weight': 'bold',
                     'axes.labelweight': 'bold',
                     'axes.titleweight': 'bold',
                     'axes.linewidth': 3,
                     'axes.grid': True,
                     'axes.grid.which': 'both',
                    'axes.spines.left': False,
                    'axes.spines.right': False,
                    'axes.spines.top': False,
                    'axes.spines.bottom': False,
                     'axes.facecolor': '#EBEBEB',
                     'xtick.color': 'black',
                     'ytick.color': 'black',
                     'ytick.minor.visible': False,
                     'xtick.minor.visible' : False,
                     'grid.color':'black',
                     'grid.linestyle': '-.',
                     'grid.linewidth': 0.6,
                     'figure.autolayout': True,
                     'figure.figsize': (19.4, 10)}

plt.rcParams.update(plot_style)
# evaluation metric name
metric_names = {
            "stats": ["avg_hops_over_time",
                          "signalling ratio",
                          "total_hops_over_time",
                          "all_signalling_ratios"
                          ], 
                "nb_new_pkts": ["pkts_over_time"],
                "nb_lost_pkts": ["pkts_over_time"],
                "nb_arrived_pkts": ["pkts_over_time"],
                f"test_results":["test_overlay_loss_rate", 
                                 "test_overlay_e2e_delay",
                                #  "test_global_loss_rate", 
                                #  "test_global_e2e_delay"
]}
metric_name = "test_overlay_loss_rate"
exps_names = []

#------------------------------------------------------------
# % set the parameters
# static parameters
traff_mats = [0, 2, 3, 1]
traff_mats = list(range(9))
# traff_mats = [0,]
sync_steps = list(range(1, 8))
sync_steps =  list(range(1, 10)) + list(range(10, 22, 2))
sync_steps = list(range(1, 10, 2))
sync_steps = [1, 5, 7, 9]

# sync_steps.remove(1)
# sync_steps.remove(3)
# sync_steps = [1,]
seed = 100
rb_size = 10000
signaling_types = ["NN", "digital_twin", "target", "sp", "opt_10"]
signaling_types = ["digital_twin", "NN"]
# signaling_types = ["NN",]
# dqn_models = ["", "_lite", "_lighter", "_lighter_2", "_lighter_3", "_ff"][:-1]
dqn_models = ["",]
nn_size = 35328
# variable parameters
training_loads = [0.4, 0.7, 0.9]
training_loads = [0.9]
topology_name = "geant"
bs = 512
lr = 0.00001
exploration = ["vary", 1.0, 0.01]
ping_freq = 10000
mv_avg_interval = 5
max_output_buffer_size = 16260
train_duration = 20
test_loads = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
loss_penalty_types = ["constrained" ,"fixed", "None"]
# loss_penalty_types = ["constrained"]
name_to_save = ["constrained", "fixed", "None"]
lambda_train_step = -1
buffer_soft_limit =0
snapshots =  ["final"]
snapshots = ["final"] + [f"episode_{i}_step_{i}" for i in range(1, 20, 1)]
ping_freqs = [0.1, 0.5, 1]
ping_freqs = [0.1,]
# snapshots =  ["final"]
            
#% reset the variables
all_exps_results = {}
# go through all the experiments
found = []
thresholds = [0.0, 0.25, 0.5, 0.75]
# thresholds = [0.0]
loss_flag = 1
use_loss_flag = 1
optimal_root_path = "/mnt/backup_examples_new/5n_overlay_full_mesh_abilene/optimal_solution/"

def load_data(path, metric_name, case):
    exp_df = convert_tb_data(path)
    if case in ["stats", "nb_new_pkts", "nb_lost_pkts", "nb_arrived_pkts"]:
        if metric_name in ("all_signalling_ratios",):
            # compute average signalling ratio per episode
            exp_df.step = pd.to_datetime(exp_df.step/1000, unit="ms")
            exp_df = exp_df.resample("20s", on="step", closed="right").mean()
            exp_df.step = exp_df.index
        else:
            exp_df = exp_df.iloc[-1]
    steps, indices = np.unique(exp_df.step, return_index=True)
    if len(steps) > 1:
        data_to_store = np.array(exp_df.value)[indices].reshape(-1, 1)
    else:
        data_to_store = np.array(exp_df.value).reshape(-1, 1)

#% Convergence 1 : plot the metrics vs episode for each loss_penalty_type, for NN model sync step 1 and train load 0.9
topology_name = "geant"
if topology_name == "abilene":
    all_exps_results = np.load("/home/redha/PRISMA_copy/prisma/all_exps_results_old.npy", allow_pickle=True).item()
    traff_mats = np.arange(7)
elif topology_name == "geant":
    # all_exps_results = np.load("/home/redha/PRISMA_copy/prisma/all_exps_results_geant.npy", allow_pickle=True).item()
    traff_mats = np.arange(5, 10)
    signaling_type = "NN"
    episodes = [f"episode_{i}_step_{i}" for i in [1, 2, 3, 4, 5, 6, 7, 10, 15, 19]]+ ["final"]
    for traff_mat in traff_mats:
        for loss_penalty_type in ["None", "fixed", "constrained"]:
            for metric_name in metric_names[f"test_results"]:
                for i, snapshot in enumerate(episodes):
                    sync_step = 1 if traff_mat < 4 else 5
                    dt_time = 5 if traff_mat < 4 else 3
                    session_name = f"sync_{sync_step}_mat_{traff_mat}_dqn__{signaling_type}_size_{nn_size}_tr_{0.9}_sim_20_20_lr_1e-05_bs_512_outb_16260_losspen_{loss_penalty_type}_lambda_step_-1_ratio_{0}_wait_{0}_lambda_lr_1e6_dt_time_{dt_time}_ping_{0.1}_vary_one_explo_first_loss_{loss_flag}_reset_{0}_use_loss_{use_loss_flag}_threshold_{0.0}"
                    ### add the data to the results
                    pth = f"/mnt/backup_examples_new/overlay_full_mesh_10n_geant/results/geant_results_with_threshold/{session_name}/test_results/{snapshot}"
                    
                    try:
                        exp_df = convert_tb_data(pth)
                    except:
                        if metric_name == "test_overlay_loss_rate":
                            print(traff_mat, loss_penalty_type, metric_name, snapshot)
                        continue
                    data = exp_df[exp_df.name == metric_name].sort_values("step")
                    steps, indices = np.unique(data.step, return_index=True)
                    if len(steps) > 1:
                        data_to_store = np.array(data.value)[indices].reshape(-1, 1)
                    else:
                        data_to_store = np.array(data.value).reshape(-1, 1)
                    key = f"{signaling_type}_1_{loss_penalty_type}_0.9_0.1_{snapshot}_test_results_{metric_name}_0.0"
                    if len(data_to_store) !=7:
                        print(key)
                        continue
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
                        
metric_names_str = ["loss", "delay"]
for metric_index, metric in enumerate(metric_names[f"test_results"]):
    plots = []
    fig, ax = plt.subplots()
    for threshold in [0.0, ]:
        for loss_penalty_type in ["None", "fixed", "constrained"]:
            # add dqn buffer NN model
            y = []
            y_min = []
            y_max = []
            x = []
            for i, snapshot in enumerate(episodes):
                name = f"{signaling_type}{''}_{1}_{loss_penalty_type}_{0.9}_{0.1}_{snapshot}_test_results"
                temp = np.mean(all_exps_results[f"{name}_{metric}_{threshold}"]["data"], axis=0)
                y.append(np.mean(temp))
                y_min.append(np.mean(temp) - np.std(temp))
                y_max.append(np.mean(temp) + np.std(temp))
                x.append(i)
            # add the plot into the figure with min and max values shaded area
            plots.append(ax.plot(x, y, label=f"{loss_penalty_type}".replace("None", "loss blind").replace("fixed", "fixed loss-pen").replace("constrained", "guided reward").replace("NN", "Model Sharing offline").replace("digital_twin", "DT"),  marker="o",
                                linewidth=7, markersize=20))
            # ax.fill_between(x, y_min, y_max, alpha=0.2)
      
        plt.xlabel(xlabel="Episode number")
        metric_name = f"{''.join(metric.split('_')[1:])}".replace("cost", "Cost").replace("lossrate", "Packet Loss Ratio").replace("delay", " Delay (ms)").replace("overlay", "").replace("global", "Global ").replace("e2e", "End-to-End ")
        plt.ylabel(ylabel=f"{metric_name}")
        if metric == "test_overlay_loss_rate":
            plt.yticks(ticks=plt.yticks()[0], labels=[f"{np.round(y*100, 2)}%" for y in plt.yticks()[0]])
            plt.ylim(0)
        else:
            plt.yticks(ax.get_yticks(), np.array(ax.get_yticks()*1000, dtype=int))
        # plt.legend(loc="upper left")
        
        # add sp and opt
        # y = []
        # for tr_idx in traff_mats:
        #     y.append(np.mean(all_exps_results[f"sp_{tr_idx}_test_results_{metric}"]["data"], axis=0)[0])
        # ax.hlines(np.mean(y), 0, 20, label="OSPF", linestyle="--", color="red", 
        #                         linewidth=7)
        # # ax.fill_between(sync_steps, np.min(y, axis=0), np.max(y, axis=0), alpha=0.2, color="red")
        # y = []
        # for tr_idx in traff_mats:
        #     y.append(np.mean(all_exps_results[f"opt_{tr_idx}_test_results_{metric}"]["data"], axis=0))
        # ax.hlines(np.mean(y), 0, 20, label="Oracle Routing", linestyle="--", color="tab:purple", linewidth=7)
        # plt.xticks(ax.get_xticks(), ax.get_xticks()+1)
        # plt.tight_layout()
    # tikzplotlib.clean_figure()
    # tikzplotlib.save(f"/home/redha/PRISMA_copy/prisma/figures/journal_paper_1/{topology_name}_{metric_names_str[metric_index]}_vs_sync_steps.tex", axis_width="\\figwidth", axis_height="\\figheight", extra_axis_parameters={"mark options={scale=2}"})
    
    # plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/q2/{topology_name}_{metric_names_str[metric_index]}_vs_episodes.pdf".replace(" ", "_"), bbox_inches='tight', pad_inches=0.1)

# add the legend
# generate empty figure to include only the legend
plt.figure()
plt.axis('off')
# remove right and top margins and paddings
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
# add legend
plt.legend(loc="upper left", handles=[bp[0] for bp in plots], ncol=3)
# shrink the figure size to fit the box
plt.gcf().set_size_inches(0.1, 0.1, forward=True)
# remove the white space around the figure
plt.tight_layout()
# plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/q2/legend.pdf".replace(" ", "_"),bbox_inches='tight', pad_inches=0.1)
# %%
