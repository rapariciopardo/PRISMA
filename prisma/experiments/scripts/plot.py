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
sys.path.append('../../source/')
from utils import convert_tb_data


# define the style of the plots
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
                                 "test_overlay_e2e_delay"
]}
#------------------------------------------------------------
# % set the parameters
# parameters
traff_mats = list(range(9))
sync_steps = [5, 7, 9]
signaling_types = ["digital_twin", "NN"]
topology_name = "geant"
test_loads = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
loss_penalty_types = ["constrained" ,"fixed", "None"]
snapshots = ["final"] + [f"episode_{i}_step_{i}" for i in range(1, 20, 1)]
thresholds = [0.0, 0.25, 0.5, 0.75]
#% reset the variables
all_exps_results = {}
# go to the directory where the results are stored
os.chdir(f"../../topologies/{topology_name}/")


#------------------------------------------------------------
#%% recover sp and opt results
for name in ["opt","sp"]:
    for traff_mat in traff_mats:
        data_loss = []
        data_delay = []
        pth = f"results/exps/logs/{name}_{traff_mat}"
        exp_df = convert_tb_data(pth)
        for metric_name in metric_names[f"test_results"]:
            data = exp_df[exp_df.name == metric_name].sort_values("step")
            steps, indices = np.unique(data.step, return_index=True)
            if len(steps) > 1:
                data_to_store = np.array(data.value)[indices].reshape(-1, 1)
            else:
                data_to_store = np.array(data.value).reshape(-1, 1)
            key = f"{name}_{traff_mat}_test_results_{metric_name}"
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
#1)------------------------------------------------------------
#%% Perf 1: plot the results & save the figures for loss rate, delay vs loads for digital twin, sp and opt
all_exps_results = np.load(f"plots/data/all_exps_results.npy", allow_pickle=True).item()
if len(all_exps_results) == 0:
    for metric_name in metric_names[f"test_results"]:
        session_name = f"sync_1_mat_{traff_mat}_dqn__digital_twin_size_35328_tr_0.9_sim_20_20_lr_1e-05_bs_512_outb_16260_losspen_constrained_lambda_step_-1_ratio_0_wait_0_lambda_lr_1e6_dt_time_3_ping_0.1_vary_one_explo_first_loss_1_reset_0_use_loss_1_threshold_0.0"
        ### add the data to the results
        pth = f"../topologies/{topology_name}/results/exps/logs/{session_name}/test_results/final"
        try:
            exp_df = convert_tb_data(pth)
        except:
            print(pth)
            continue
        data = exp_df[exp_df.name == metric_name].sort_values("step")
        steps, indices = np.unique(data.step, return_index=True)
        if len(steps) > 1:
            data_to_store = np.array(data.value)[indices].reshape(-1, 1)
        else:
            data_to_store = np.array(data.value).reshape(-1, 1)
        key = f"digital_twin_1_constrained_0.9_0.1_final_test_results_{metric_name}_0.0"
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
            
            
model_names = ["O-DQR", "ISP default path", "Oracle Routing"]
colors = ["blue", "red", "green"]
load_to_plot = [60, 90, 120]
for metric in metric_names["test_results"]:
    fig, ax = plt.subplots()
    bps = []
    data = all_exps_results[f"digital_twin_1_constrained_0.9_0.1_final_test_results_{metric}_0.0"]
    x = data["loads"]
    y_dt = np.mean(data["data"], axis=1)
    # add sp and opt
    y_sp = np.array([])
    for i, tr_idx in enumerate(traff_mats):
        data = all_exps_results[f"sp_{tr_idx}_test_results_{metric}"]["data"]
        if i == 0:
            y_sp = data
        else:
            y_sp = np.concatenate((y_sp, data), axis=1)
    y_sp = np.mean(y_sp, axis=1)
    y_opt = np.array([])
    for i, tr_idx in enumerate(traff_mats):
        data = all_exps_results[f"opt_{tr_idx}_test_results_{metric}"]["data"]
        if i == 0:
            y_opt = data
        else:
            y_opt = np.concatenate((y_opt, data), axis=1)
    y_opt = np.mean(y_opt, axis=1)
    for idx, cat in enumerate(["low", "medium", "high"]):
        if cat == "low":
            bps.append(plt.bar(idx-0.2, y_opt[x.index(load_to_plot[idx])], width=0.2, color=colors[2], label=model_names[2]))
            bps.append(plt.bar(idx, y_dt[x.index(load_to_plot[idx])], width=0.2, color=colors[0], label=model_names[0]))
            bps.append(plt.bar(idx+0.2, y_sp[x.index(load_to_plot[idx])], width=0.2, color=colors[1], label=model_names[1]))
            
        elif cat == "medium":
            bps.append(plt.bar(idx-0.2, y_opt[x.index(load_to_plot[idx])], width=0.2, color=colors[2]))
            bps.append(plt.bar(idx, y_dt[x.index(load_to_plot[idx])], width=0.2, color=colors[0]))
            bps.append(plt.bar(idx+0.2, y_sp[x.index(load_to_plot[idx])], width=0.2, color=colors[1]))
        else:
            bps.append(plt.bar(idx-0.2, y_opt[x.index(load_to_plot[idx])], width=0.2, color=colors[2]))
            bps.append(plt.bar(idx, y_dt[x.index(load_to_plot[idx])], width=0.2, color=colors[0]))
            bps.append(plt.bar(idx+0.2, y_sp[x.index(load_to_plot[idx])], width=0.2, color=colors[1]))

    plt.xlabel(xlabel="Traffic Load")
    plt.xticks(ticks=[0, 1, 2], labels=["Low", "Mid", "High"])
    if metric == "test_overlay_loss_rate":
        plt.ylabel(ylabel="Packet Loss Ratio")
        plt.yticks(ticks=plt.yticks()[0], labels=[f"{int(y*100)}%" for y in plt.yticks()[0]])
    if metric == "test_overlay_e2e_delay":
        plt.ylabel(ylabel="Delay (ms)")
        plt.ylim(0.45, 0.7)
        plt.yticks(ticks=plt.yticks()[0], labels=[f"{int(y*1000)}" for y in plt.yticks()[0]])
    print(f"DT: {y_dt[x.index(load_to_plot[0])]}, {y_dt[x.index(load_to_plot[1])]}, {y_dt[x.index(load_to_plot[2])]}")
    print(f"SP: {y_sp[x.index(load_to_plot[0])]}, {y_sp[x.index(load_to_plot[1])]}, {y_sp[x.index(load_to_plot[2])]}")
    print(f"OPT: {y_opt[x.index(load_to_plot[0])]}, {y_opt[x.index(load_to_plot[1])]}, {y_opt[x.index(load_to_plot[2])]}")
    # save the figure
    plt.savefig(f"plots/q1/{topology_name}_{metric}_vs_loads_.png".replace(" ", "_"),bbox_inches='tight', pad_inches=0.1)

# generate empty figure to include only the legend
plt.figure()
plt.axis('off')
# remove right and top margins and paddings
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
# add legend
plt.legend(loc="upper left", handles=[bp for bp in bps], ncol=3)
# shrink the figure size to fit the box
plt.gcf().set_size_inches(0.1, 0.1, forward=True)
# remove the white space around the figure
plt.tight_layout()
plt.savefig(f"plots/q1/legend_.png".replace(" ", "_"),bbox_inches='tight', pad_inches=0.1)
                            
#2)------------------------------------------------------------
#%% Convergence 1 : plot the metrics vs episode for each loss_penalty_type, for NN model sync step 1 and train load 0.9
topology_name = "abilene"
if topology_name == "abilene":
    all_exps_results = np.load("/home/redha/PRISMA_copy/prisma/all_exps_results_old.npy", allow_pickle=True).item()
    traff_mats = np.arange(11)
elif topology_name == "geant":
    # all_exps_results = np.load("/home/redha/PRISMA_copy/prisma/all_exps_results_geant.npy", allow_pickle=True).item()
    traff_mats = np.arange(9)
    signaling_type = "digital_twin"
    episodes = [f"episode_{i}_step_{i}" for i in range(1, 20, 3)]+ ["final"]
    for traff_mat in traff_mats:
        for loss_penalty_type in ["None", "fixed", "constrained"]:
            for metric_name in metric_names[f"test_results"]:
                for i, snapshot in enumerate(episodes):
                    session_name = f"sync_1_mat_{traff_mat}_dqn__{signaling_type}_size_{35328}_tr_{0.9}_sim_20_20_lr_1e-05_bs_512_outb_16260_losspen_{loss_penalty_type}_lambda_step_-1_ratio_{0}_wait_{0}_lambda_lr_1e6_dt_time_3_ping_{0.1}_vary_one_explo_first_loss_{1}_reset_{0}_use_loss_{1}_threshold_{0.0}"
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
    
    plt.savefig(f"plots/q2/{topology_name}_{metric_names_str[metric_index]}_vs_episodes.png".replace(" ", "_"), bbox_inches='tight', pad_inches=0.1)

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
plt.savefig(f"plots/q2/legend.png".replace(" ", "_"),bbox_inches='tight', pad_inches=0.1)
    
#3)------------------------------------------------------------
#%% Overhead 1 : plot the results & save the figures for overhead vs sync step

topology_name = "geant"
if topology_name == "abilene":
    all_exps_results = np.load("/home/redha/PRISMA_copy/prisma/all_exps_results_abilene.npy", allow_pickle=True).item()
    all_exps_results_old = np.load("/home/redha/PRISMA_copy/prisma/all_exps_results_old.npy", allow_pickle=True).item()
    nb_links = 4*5
elif topology_name == "geant":
    # all_exps_results = np.load("/home/redha/PRISMA_copy/prisma/all_exps_results_geant.npy", allow_pickle=True).item()
    # all_exps_results_old = np.load("/home/redha/PRISMA_copy/prisma/all_exps_results_geant.npy", allow_pickle=True).item()
    nb_links = 10*9
    traff_mats = np.arange(9)
    signaling_type = "digital_twin"
    episodes = [f"episode_{i}_step_{i}" for i in range(1, 20, 3)]+ ["final"]
    loss_penalty_type = "constrained"
    snapshot = "final"
    for traff_mat in traff_mats:
            ## load digital twin test results
            signaling_type = "digital_twin"
            for threshold in [0.0, 0.25, 0.5, 0.75]:
                session_name = f"sync_1_mat_{traff_mat}_dqn__{signaling_type}_size_{nn_size}_tr_{0.9}_sim_20_20_lr_1e-05_bs_512_outb_16260_losspen_{loss_penalty_type}_lambda_step_-1_ratio_{0}_wait_{0}_lambda_lr_1e6_dt_time_3_ping_{0.1}_vary_one_explo_first_loss_{loss_flag}_reset_{0}_use_loss_{use_loss_flag}_threshold_{0.0}"
                for metric_name in metric_names[f"test_results"]:
                    ### add the data to the results
                    pth = f"/mnt/backup_examples_new/overlay_full_mesh_10n_geant/results/geant_results_with_threshold/{session_name}/test_results/{snapshot}"
                    try:
                        exp_df = convert_tb_data(pth)
                    except:
                        if metric_name == "test_overlay_loss_rate":
                            print(traff_mat, loss_penalty_type, metric_name, snapshot, session_name)
                        continue
                    data = exp_df[exp_df.name == metric_name].sort_values("step")
                    steps, indices = np.unique(data.step, return_index=True)
                    if len(steps) > 1:
                        data_to_store = np.array(data.value)[indices].reshape(-1, 1)
                    else:
                        data_to_store = np.array(data.value).reshape(-1, 1)
                    key = f"{signaling_type}_1_{loss_penalty_type}_0.9_0.1_{snapshot}_test_results_{metric_name}_{threshold}"
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
                ### signaling ratio and nb new pkts
                for metric_name in ["signalling ratio", "nb new pkts"]:
                    if metric_name == "signalling ratio":
                        pth = f"/mnt/backup_examples_new/overlay_full_mesh_10n_geant/results/geant_results_with_threshold/{session_name}/stats"
                        try:
                            exp_df = convert_tb_data(pth)
                        except:
                            print("stats", traff_mat, loss_penalty_type, metric_name, session_name)
                            continue
                        data = exp_df[exp_df.name == metric_name].sort_values("step")
                        # compute average signalling ratio per episode
                        data.step = pd.to_datetime(data.step/1000, unit="ms")
                        data = data.resample("20s", on="step", closed="right").mean()
                        data.step = data.index
                    else:
                        pth = f"/mnt/backup_examples_new/overlay_full_mesh_10n_geant/results/geant_results_with_threshold/{session_name}/new_pkts"
                        try:
                            exp_df = convert_tb_data(pth)
                        except:
                            print("new_pkts", traff_mat, loss_penalty_type, metric_name, session_name)
                            continue
                        data = exp_df[exp_df.name == metric_name].sort_values("step")
                        data = data.iloc[-1]
                    steps, indices = np.unique(data.step, return_index=True)
                    if len(steps) > 1:
                        data_to_store = np.array(data.value)[indices].reshape(-1, 1)
                    else:
                        data_to_store = np.array(data.value).reshape(-1, 1)
                    key = f"{signaling_type}_1_{loss_penalty_type}_0.9_0.1_{snapshot}_final_stats_{metric_name}_{threshold}"
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
            
            ### load NN test results
            threshold = 0.0
            signaling_type = "NN"
            for sync_step in [5, 7, 9]:
                for metric_name in metric_names[f"test_results"]:
                        if traff_mat < 4:
                            dt_nn = 5
                        else:
                            dt_nn = 3
                        session_name = f"sync_{sync_step}_mat_{traff_mat}_dqn__{signaling_type}_size_{nn_size}_tr_{0.9}_sim_20_20_lr_1e-05_bs_512_outb_16260_losspen_{loss_penalty_type}_lambda_step_-1_ratio_{0}_wait_{0}_lambda_lr_1e6_dt_time_{dt_nn}_ping_{0.1}_vary_one_explo_first_loss_{loss_flag}_reset_{0}_use_loss_{use_loss_flag}_threshold_{0.0}"
                        ### add the data to the results
                        pth = f"/mnt/backup_examples_new/overlay_full_mesh_10n_geant/results/geant_results_with_threshold/{session_name}/test_results/{snapshot}"
                        try:
                            exp_df = convert_tb_data(pth)
                        except:
                            if metric_name == "test_overlay_loss_rate":
                                print("test", traff_mat, loss_penalty_type, metric_name, snapshot, session_name)
                            continue
                        data = exp_df[exp_df.name == metric_name].sort_values("step")
                        steps, indices = np.unique(data.step, return_index=True)
                        if len(steps) > 1:
                            data_to_store = np.array(data.value)[indices].reshape(-1, 1)
                        else:
                            data_to_store = np.array(data.value).reshape(-1, 1)
                        key = f"{signaling_type}_{sync_step}_{loss_penalty_type}_0.9_0.1_{snapshot}_test_results_{metric_name}_0.0"
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
                ### signaling ratio and nb new pkts
                for metric_name in ["signalling ratio", "nb new pkts"]:
                    if metric_name == "signalling ratio":
                        pth = f"/mnt/backup_examples_new/overlay_full_mesh_10n_geant/results/geant_results_with_threshold/{session_name}/stats"
                        try:
                            exp_df = convert_tb_data(pth)
                        except:
                            print("stats", traff_mat, loss_penalty_type, metric_name, session_name)
                            continue
                        # compute average signalling ratio per episode
                        data = exp_df[exp_df.name == metric_name].sort_values("step")
                        data.step = pd.to_datetime(data.step/1000, unit="ms")
                        data = data.resample("20s", on="step", closed="right").mean()
                        data.step = data.index
                    else:
                        pth = f"/mnt/backup_examples_new/overlay_full_mesh_10n_geant/results/geant_results_with_threshold/{session_name}/new_pkts"
                        try:
                            exp_df = convert_tb_data(pth)
                        except:
                            print("new_pkts", traff_mat, loss_penalty_type, metric_name, session_name)
                            continue
                        data = exp_df[exp_df.name == metric_name].sort_values("step")
                        data = data.iloc[-1]
                    steps, indices = np.unique(data.step, return_index=True)
                    if len(steps) > 1:
                        data_to_store = np.array(data.value)[indices].reshape(-1, 1)
                    else:
                        data_to_store = np.array(data.value).reshape(-1, 1)
                    key = f"{signaling_type}_1_{loss_penalty_type}_0.9_0.1_{snapshot}_final_stats_{metric_name}_{threshold}"
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
                
                
overheads =  {}
metric_names_str = ["loss", "delay"]

costs =  {"test_overlay_e2e_delay_NN_0.0": [],
        "test_overlay_loss_rate_NN_0.0": []}
fig, ax = plt.subplots()
# add dqn buffer NN model
for signaling_type in ["NN", "digital_twin"]:
    for loss_penalty_type in ["constrained"]:
        for threshold in [0.0, 0.25, 0.5, 0.75]:
                for training_load in [0.9]:
                    y = []
                    y_min = []
                    y_max = []
                    x = []
                    if signaling_type != "digital_twin" and threshold > 0.0:
                        continue
                    for sync_step in [1, 3, 5, 7, 9]:
                        if signaling_type != "NN":
                            temp = all_exps_results[f"{signaling_type}_{1}_{loss_penalty_type}_{training_load}_{0.1}_final_stats_signalling ratio_{threshold}"]['data'].max()
                            print("max", all_exps_results[f"{signaling_type}_{1}_{loss_penalty_type}_{training_load}_{0.1}_final_stats_signalling ratio_{threshold}"]['data'])

                            costs[f"test_overlay_e2e_delay_{signaling_type}_{threshold}"] = np.mean(all_exps_results[f"{signaling_type}_{1}_{loss_penalty_type}_{0.9}_{0.1}_final_test_results_test_overlay_e2e_delay_{threshold}"]['data'][[0, 3, 6], :] )
                            costs[f"test_overlay_loss_rate_{signaling_type}_{threshold}"] = np.mean(all_exps_results[f"{signaling_type}_{1}_{loss_penalty_type}_{0.9}_{0.1}_final_test_results_test_overlay_loss_rate_{threshold}"]['data'][[0, 3, 6], :] )
                        else:
                            temp = all_exps_results[f"{signaling_type}_{sync_step}_{loss_penalty_type}_{training_load}_{0.1}_final_stats_signalling ratio_{threshold}"]['data'].max() + ((69*nb_links*400/sync_step)/(all_exps_results[f"{signaling_type}_{sync_step}_{loss_penalty_type}_{training_load}_{0.1}_final_nb_new_pkts_pkts_over_time_{threshold}"]['data'].max()*20))
                            
                            costs[f"test_overlay_e2e_delay_{signaling_type}_{threshold}"].append(np.mean(all_exps_results[f"{signaling_type}_{sync_step}_{loss_penalty_type}_{0.9}_{0.1}_final_test_results_test_overlay_e2e_delay_{threshold}"]['data'][[0, 3, 6], :] ))
                            costs[f"test_overlay_loss_rate_{signaling_type}_{threshold}"].append(np.mean(all_exps_results[f"{signaling_type}_{sync_step}_{loss_penalty_type}_{0.9}_{0.1}_final_test_results_test_overlay_loss_rate_{threshold}"]['data'][[0, 3, 6], :] ))
                        y.append(np.mean(temp))
                        y_min.append(np.mean(temp) - np.std(temp))
                        y_max.append(np.mean(temp) + np.std(temp))
                        x.append(sync_step)
                    # add the plot into the figure with min and max values shaded area
                    overheads[f"{signaling_type}_{threshold}"] = y
                    ax.plot(x, y, label=f"{signaling_type}_{threshold}".replace("NN", "Model Sharing").replace("digital_twin", "Digital Twin").replace("target", "Target Value Sharing").replace("sp", "OSPF").replace("opt", "Oracle Routing"), marker="o",
                                linewidth=7, markersize=20)
                    ax.fill_between(x, y_min, y_max, alpha=0.2)
# add digital twin
# temp = (all_exps_results["digital_twin" + "" + f"_{1}_5_overlay_big_signalling_bytes"]["data"] + all_exps_results["digital_twin" + "" + f"_{1}_5_overlay_small_signalling_bytes"]["data"])/all_exps_results["NN" + "" + f"_{1}_5_overlay_data_pkts_injected_bytes_time"]["data"]
# plt.hlines(np.mean(temp), 1, 8, label="DT", linestyle="--")
#add target
# temp = (all_exps_results["target" + "" + f"_{1}_5_overlay_big_signalling_bytes"]["data"] + all_exps_results["target" + "" + f"_{1}_5_overlay_small_signalling_bytes"]["data"])/all_exps_results["NN" + "" + f"_{1}_5_overlay_data_pkts_injected_bytes_time"]["data"]
# plt.hlines(np.mean(temp), 1, 8, label=f"target", linestyle="--", color="grey")
plt.xlabel(xlabel="Sync Step")
plt.ylabel(ylabel=f"Overhead Ratio")
ax.set_xticks(sync_steps)
# list(np.round(np.unique(overheads['NN']), 1)) + list(np.round(np.unique(overheads['digital_twin']), 1)) + list(np.round(np.unique(overheads['target']), 1)))
# ax.set_yticks([0.3, 1,1.5, 2, 3, 4, 5.7])
plt.legend()
# plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/final/overhead_ratio_vs_sync_steps.pdf",bbox_inches='tight', pad_inches=0.1)
#%6)------------------------------------------------------------

# plot the results & save the figures for cost vs overhead ratio
thresholds = [0.0, 0.25, 0.5, 0.75]
min_x =0.08
max_x = 2.4
max_data = 3
sync_steps = [1, 3, 5, 7, 9]
for metric_index, metric in enumerate(metric_names[f"test_results"]):
    fig, ax = plt.subplots(figsize=(18, 14))
    y = []
    x = []
    value_dt = []
    for threshold in thresholds:
        # add dqn buffer NN model
        if threshold == 0.0:
            print("nn", metric, threshold, len(costs[f"{metric}_NN_{threshold}"]), len(overheads[f"NN_{threshold}"]))
            nn_plot = ax.plot(overheads[f"NN_{threshold}"][1:], costs[f"{metric}_NN_{threshold}"][1:], label=f"Model-Sharing", marker="o", linewidth=7, markersize=20, color="tab:purple")
            # write the values
            for i, val in enumerate(costs[f"{metric}_NN_{threshold}"]):
                if sync_steps[i] in  (1,3):
                    continue
                inc_x = -0.028
                if sync_steps[i] == 9 and metric == "test_overlay_loss_rate":
                    inc_x = -0.05
                ax.text(overheads[f"NN_{threshold}"][:][i]+inc_x, val+0.0007, f"U={sync_steps[i]}s", fontsize=35, rotation=0, color="tab:purple", fontweight="bold")

            
        # add digital twin
        # if threshold == 0.0:
        #     dt_plot = ax.plot(overheads[f"digital_twin_{threshold}"][0], costs[f"{metric}_digital_twin_{threshold}"], label=f"Logit Sharing", marker="*", markersize=30, linewidth=0, color="green")
        # else:
        #     dt_plot = ax.plot(overheads[f"digital_twin_{threshold}"][0], costs[f"{metric}_digital_twin_{threshold}"], marker="*", markersize=30, linewidth=0, color="green")
        # print("dt", metric, threshold, np.mean(costs[f"{metric}_digital_twin_{threshold}"]))
        # if threshold == 0.0:
        value_dt.append(np.mean(costs[f"{metric}_digital_twin_{threshold}"]))
        y.append(costs[f"{metric}_digital_twin_{threshold}"])
        x.append(overheads[f"digital_twin_{threshold}"][0])
        # x.append(np.mean(all_exps_results[f"digital_twin_1_constrained_0.9_final_stats_all_signalling_ratios_{threshold}"]["data"], axis=1)[-1])
        # y.append(costs[f"{metric}_digital_twin_{threshold}"])
        pos_x = overheads[f"digital_twin_{threshold}"][0]
        pos_y = costs[f"{metric}_digital_twin_{threshold}"]
        if metric == "test_overlay_loss_rate":
            if threshold == 0.0:
                pos_x -= 0.05
                pos_y -= 0.003
            if threshold == 0.25:
                pos_x -= 0.20
                pos_y -= 0.002
            if threshold == 0.75:
                pos_x -= 0.25
                pos_y -= 0.002
        else:
            if threshold == 0.0:
                pos_x += 0.03
                pos_y += 0.000
            if threshold == 0.25:
                pos_y -= 0.000
                pos_x += 0.02
            if threshold == 0.75:
                pos_y -= 0.00
        ax.text(pos_x, pos_y, f"  ER_thr = {int(threshold*100)}%", fontsize=35, rotation=0, color="blue", fontweight="bold")
        # add target

        # ax.plot(overheads[f"target_{0.0}"][0], costs[f"{metric}_target_{0.25}"], label=f"Target Value Sharing {0.25}", marker="*", markersize=30, linewidth=0,)
    our_model_plot = ax.plot(x, y, label=f"Our model", marker="*", markersize=30, color="blue", linewidth=7)
    # add sp and opt
    sp_y = []
    for tr_idx in traff_mats:
        sp_y.append(np.mean(all_exps_results[f"sp_{tr_idx}_test_results_{metric}"]["data"][[0, 3, 6]], axis=0)[0])
    sp_plot = ax.hlines(np.mean(sp_y), min_x, max_x, label="Shortest Path", linestyle="--", color="red", 
                             linewidth=7)
    # ax.fill_between(sync_steps, np.min(y, axis=0), np.max(y, axis=0), alpha=0.2, color="red")
    opt_y = []
    # for tr_idx in [0, 1, 2, 3, 4,  5, 6, 8]:
    for tr_idx in traff_mats:
        opt_y.append(np.mean(all_exps_results[f"opt_{tr_idx}_test_results_{metric}"]["data"][[0, 3, 6]], axis=0)[0])
    print("opt", metric, np.mean(opt_y), opt_y)
    for i, th in enumerate(thresholds):
        print("ratio sp", th, metric, (1-(value_dt/np.mean(sp_y))))
        print("ratio opt", th, metric, (value_dt[i]-np.mean(opt_y))/(np.mean(opt_y)+value_dt[i]), opt_y)
    opt_plot = ax.hlines(np.mean(opt_y), 0, 6, label="Oracle Routing", linestyle="--", color="green", linewidth=7)

    plt.xlabel(xlabel="Maximum Overhead Ratio during Training")
    metric_name = f"{''.join(metric.split('_')[1:])}".replace("cost", "Cost").replace("lossrate", "Loss Rate").replace("delay", " Delay").replace("overlay", "Overlay ").replace("global", "Global ")
    plt.xlim(min_x, max_x)
    ticks = [0.15, 0.75, 1.0, 1.2, 0.39, 0.54, 0.30, 1.42, 1.70, 2.0, 2.17]
    ax.set_xticks(ticks, [f"{int(x*100)}%" for x in ticks], rotation=90, fontsize=40)
    if metric == "test_overlay_loss_rate":
        plt.ylabel(ylabel="Packet Loss Ratio")
        ax.set_yticks(plt.yticks()[0], [f"{np.round(x*100, 2)}%" for x in plt.yticks()[0]])
        plt.ylim(0, 0.03)
    else:
        plt.ylabel(ylabel="Delay (ms)")
        ax.set_yticks(plt.yticks()[0], [f"{int(x*1000)}" for x in plt.yticks()[0]])
        # plt.ylim(0.56)
    # ax.legend( loc="upper right")
    plt.tight_layout()
    # tikzplotlib.clean_figure()
    # tikzplotlib.save(f"/home/redha/PRISMA_copy/prisma/figures/journal_paper_1/{topology_name}_{metric_names_str[metric_index]}_vs_overhead.tex", axis_width="\\figwidth", axis_height="\\figheight", extra_axis_parameters={"mark options={scale=2}"})

    # plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/q3/{topology_name}_{metric_names_str[metric_index]}_vs_overhead.pdf".replace(" ", "_"),bbox_inches='tight', pad_inches=0.1)
    
# add legend 3
# generate empty figure to include only the legend
plt.figure(figsize=(19.4, 10))
plt.axis('off')
# remove right and top margins and paddings
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.legend(nn_plot+ our_model_plot + [sp_plot, opt_plot], ["Model Sharing", "O-DQR", "ISP default path", "Oracle Routing"], ncols=6)
# shrink the figure size to fit the box
plt.gcf().set_size_inches(0.1, 0.1, forward=True)
# remove the white space around the figure
# plt.tight_layout()
# plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/q3/legend_3.pdf".replace(" ", "_"),bbox_inches='tight', pad_inches=0.1)

#7)------------------------------------------------------------
#%% plot the overhead ratio vs episode for DT model sync step 1, train load 0.9, and rcpo loss penalty for each threshold
x = np.arange(1, 21, 1)
loss_penalty_type = "constrained"
metric = "all_signalling_ratios"
fig, ax = plt.subplots()
plots = []
for threshold in thresholds:
    # add dqn buffer NN model
    name = f"digital_twin{''}_{1}_{loss_penalty_type}_{0.9}_0.1_final_stats"
    y= np.mean(all_exps_results_[f"{name}_{metric}_{threshold}"]["data"], axis=1)
    print(all_exps_results[f"{name}_{metric}_{threshold}"]["data"])
    # y_min.append(np.mean(temp) - np.std(temp))
    # y_max.append(np.mean(temp) + np.std(temp))
    # x.append(i)
    # add the plot into the figure with min and max values shaded area
    plots.append(ax.plot(x, y, label=f"DT {threshold}".replace("None", "Loss Blind").replace("fixed", "Loss Aware").replace("constrained", "RCPO"),  marker="o",
                            linewidth=7, markersize=20)[0])
    # ax.fill_between(x, y_min, y_max, alpha=0.2)
plt.xlabel(xlabel="Episode Number")
metric_name = f"{''.join(metric.split('_')[1:])}".replace("cost", "Cost").replace("lossrate", "Loss Rate").replace("delay", " Delay").replace("overlay", "Overlay ").replace("global", "Global ")
# plt.ylim(0, 0.48)
plt.yticks(plt.yticks()[0], [f"{np.int(x*100)}%" for x in plt.yticks()[0]])
plt.xticks(np.arange(1, 21, 6))
plt.ylabel(ylabel=f"Max Overhead Ratio\n during Training")
plt.tight_layout()
# tikzplotlib.clean_figure()
# tikzplotlib.save(f"/home/redha/PRISMA_copy/prisma/figures/journal_paper_1/overhead_vs_episodes.tex", axis_width="\\figwidth", axis_height="\\figheight", extra_axis_parameters={"mark options={scale=2}"})

plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/q3/{topology_name}_overhead_vs_episodes.pdf".replace(" ", "_"),bbox_inches='tight', pad_inches=0.1)

# add legend 4 
# generate empty figure to include only the legend
plt.figure(figsize=(19.4, 10))
plt.axis('off')
# remove right and top margins and paddings
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.legend(plots, ["ER_thr=0%", "ER_thr=25%", "ER_thr=50%", "ER_thr=75%"], ncols=4)
# shrink the figure size to fit the box
plt.gcf().set_size_inches(0.1, 0.1, forward=True)
# remove the white space around the figure
# plt.tight_layout()
# plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/q3/{topology_name}_legend_.pdf".replace(" ", "_"),bbox_inches='tight', pad_inches=0.1)
#3)------------------------------------------------------------
#%% plot the results & save the figures for cost, loss and delay vs load factors 
figures_dir = f"/home/redha/PRISMA_copy/prisma/figures"
threshold = 0.0 
tr_mat = 6
for sync_step in [1]:
    # add dqn buffer NN model
    for metric in metric_names[f"test_results"]: 
        fig, ax = plt.subplots()
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
                            name = f"{signaling_type}{''}_{sync_step}_{loss_penalty_type}_{training_load}_{0.1}_{snapshot}_test_results"
                            # temp = np.mean(all_exps_results[f"{name}_{metric}"]["data"], axis=1)
                            temp = all_exps_results[f"{name}_{metric}_{threshold}"]["data"][:, 1]
                            y.append(np.mean(temp))
                            y_min.append(np.mean(temp) - np.std(temp))
                            y_max.append(np.mean(temp) + np.std(temp))
                            x.append(sync_step)
                            # add the plot into the figure with min and max values shaded area
                            ax.plot(all_exps_results[f"{name}_{metric}_{threshold}"]["loads"], temp,
                                    label=f"{loss_penalty_type}".replace("None", "Loss Blind").replace("fixed", "Loss Aware").replace("constrained", "RCPO"), linestyle="dotted",
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
        for name in ["sp", "opt" ]:
        # for name in ["sp", "opt_9", ]:
            y = []
            for tr_idx in traff_mats:
                y.append(all_exps_results[f"{name}_{tr_idx}_test_results_{metric}"]["data"][:, 0])
            ax.plot(all_exps_results[f"{name}_{tr_idx}_test_results_{metric}"]["loads"], np.mean(y, axis=0), label=f"{name}".replace("sp", "OSPF").replace("opt", "Oracle Routing"), linestyle="--", marker="o", linewidth=7, markersize=20, )
            print(metric, name, np.mean(np.array(y)[:, :-1]))
        # y = all_exps_results[f"opt_0_{metric}"]["data"]
        # ax.plot(all_exps_results[f"sp_0_{metric}"]["loads"], y, label="OPT", linestyle="--", color="green")
        metric_name = f"{''.join(metric.split('_')[1:])}".replace("cost", "Cost").replace("lossrate", "Loss Rate").replace("delay", " Delay").replace("overlay", "Overlay ").replace("global", "Global ")
        plt.xlabel(xlabel="Load Factor")
        plt.xticks(ticks=[60, 70, 80, 90, 100, 110, 120], labels=["60%", "70%", "80%", "90%", "100%", "110%", "120%"])
        plt.ylabel(ylabel=f"{metric_name}")
        # plt.legend(loc="upper left")
        # plt.title(f"Variation of {metric_name} with load factor in Offband setting")
        # plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/final/{metric_name}_vs_loads.pdf".replace(" ", "_"),bbox_inches='tight', pad_inches=0.1)
    












# Additional plots
#1)------------------------------------------------------------
#%% plot geant topology and tag overlay nodes
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# load the topology from adjacency matrix
G = nx.from_numpy_matrix(np.loadtxt("/home/redha/PRISMA_copy/prisma/examples/overlay_full_mesh_10n_geant/topology_files/physical_adjacency_matrix.txt"))
# nodes_coords = np.loadtxt("/home/redha/PRISMA_copy/prisma/examples/overlay_full_mesh_10n_geant/topology_files/node_coordinates.txt")
overlay_nodes = np.loadtxt("/home/redha/PRISMA_copy/prisma/examples/overlay_full_mesh_10n_geant/topology_files/map_overlay.txt")
nx.draw(G, node_size=100, edge_color="grey", width=1, with_labels=True, font_size=10, labels={node: node for node in G.nodes()}, pos=nx.kamada_kawai_layout(G))

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