# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:16:40 2022

@author: Redha

Plot for ideal, nn and target, the delay, loss and cost over load for each sync step
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.rcParams.update({'font.family': "Times New Roman"})
plt.rcParams.update({'font.size': 26})
# plt.rcParams.update({'font.serif': ["Times"]})
if __name__ == '__main__':
    ### define params
    stat_names = ["Average Loss Rate Over Loads", 
                  "Average Delay Per Arrived Pkts",
                  "Theoretical Cost",
                  "Average Cost Per Packets",
                  "Total cost", 
                  "Signaling Overhead in Bytes"]
    stat_file_names = ["loss", "delay", "rew", "real_rew", "total_rew", "overhead"]
    folder_path = "D:/NetSim_/NetSim/prisma/"
    
    kkk = 3
    """ 1 ==  sync step variation for ideal case 
        2 == signaling type comparaison for sync step 1s and 20% ratio 
        3 == ratio comparaison  for NN signaling 
    """
    model_names = ["DQN Routing - Model Sharing", "DQN Routing - Value Sharing", "Shortest Path Routing", "LP - Min Cost MCF"]
    folder_name =  f"tests_sync_variation_mat0_rb_10k_plots"
    line_styles = ["solid" , "solid", "dashed", "dashed"]
    colors = [ "purple", "green", "red","blue"]
    avg_data = {}
    test_train = ["tests_3", "train_3", "train_3", "train_3"]
    signaling_inband=0
    ### check if folder exists
    if folder_name not in os.listdir(folder_path):
        os.mkdir(folder_path + "\\" + folder_name)
    for signaling_inband in (0, 1):
        for kkk_idx, kkk in enumerate((3, 4, 3, 5)):
            # fig1, ax1 = plt.subplots()
            # fig1.set_size_inches(19.4, 10)
            
            for xxx, signaling_type in enumerate(["NN", "target"]):
                # if kkk_idx==1 and xxx >1:
                    # continue
                for stat_idx in range(len(stat_names)):
                    if stat_idx != kkk:
                        continue
                    ## stat over sync step
                    # fig1 = plt.figure()
                    # fig1.set_size_inches(19.4, 10)
                    sync_steps = []
                    
                    
                    # fig = plt.figure()
                    # fig.set_size_inches(19.4, 10)
                    list_charges = [[60, 70, 80, 90, 100, 110, 120, 130, 140], [40], [40], [40]]
                    # list_charges = [40]
            
            
                    syncs = np.arange(500, 5000, 500).tolist() + np.arange(5000, 15000, 1000).tolist() 
                    # syncs = np.arange(500, 5000, 500).tolist() + np.arange(5000, 15000, 5000).tolist()
                    syncs.remove(4500)
                    if signaling_type in ("ideal", "NN", "target"):
                        names = [f"dqn_buffer_{signaling_type}_{signaling_inband}_fixed_sync{xx}ms_ratio_{(10, 20,20, 20)[kkk_idx]}" for xx in syncs] 
                        official_names = [f"DQN Buffer {signaling_type} sync {xx}s" for xx in np.array(syncs)/1000] 
                
                    elif signaling_type == "sp":
                        names = ["sp_ideal_0_fixed_sync2500ms_ratio_10"]* len(syncs)
                        official_names = ["Shortest Path"]* len(syncs)
                    else:
                        names = ["opt_ideal_0_fixed_sync2500ms_ratio_10"] * len(syncs) 
                        official_names = ["Optimal Solution"] * len(syncs)
                    j = 0
                    for i in range(len(names)):
                        for charge_index, charge in enumerate(list_charges[kkk_idx]):
                            temp = np.loadtxt(f"D:\\NetSim_\\NetSim\\prisma\\examples\\abilene\\{test_train[kkk_idx]}\\{names[i]}_load_{charge}.txt", delimiter=',', dtype=object)
                            if len(temp.shape) == 1:
                                delay_temp = np.array(temp[6 + stat_idx], dtype=float).reshape(1, -1)
                            else:
                                delay_temp = np.array(temp[0, 6 + stat_idx], dtype=float).reshape(1, -1)
                    
                            if charge_index == 0:
                                delay = delay_temp.reshape(1, -1)
                            else:
                                delay = np.concatenate((delay, delay_temp.reshape(1, -1)))
                        print(delay.shape)
                        sync_steps.append(np.mean(delay, axis=0))
        
                        j += 1
                    print(stat_names[stat_idx], sync_steps)
                    
        

                    avg_data[f"{signaling_type}_{stat_file_names[kkk]}_{signaling_inband}_{test_train[kkk_idx]}"] = np.array(sync_steps)[:, 0]
                        
                    # ax1.plot(np.array(syncs)/1000, avg_data[f"{signaling_type}_{kkk}_{signaling_inband}"] , label=f"{model_names[xxx]}", color=colors[xxx], linestyle=line_styles[xxx], marker="o")
        
                    
            # plt.vlines(min_x_value, 0, 3.5, linestyles="dotted", label="Minimum value")
            # ax1.set_ylabel(f"{stat_names[kkk]}")
            # ax1.set_xlabel(f"Synchronisation Period T_s (s)")
            # ax1.set_xlim(0, 21)
            # ax1.set_ylim(0, 3.5)
            # ax1.set_xticks(np.arange(0, 25, 5).tolist() + [min_x_value], np.arange(0, 25, 5).tolist() + [min_x_value] )
            
            # ax2 = ax1.twinx()
            # ax2.set_yscale('linear')
            # mse = (avg_data["ideal"] - avg_data["NN"])**2
            # ax2.plot(np.array(syncs)/1000, mse, color="green", linestyle=line_styles[0], marker="o", label="MSE Between No Signaling and Off-band Signaling")
            # ax2.set_ylim(0, np.max(mse))
            # ax2.set_ylabel(f"Mean Squared Error", color="green")
            # ax2.tick_params(axis='y', labelcolor="green")
        
            # fig1.legend(loc=2)
            # fig1.tight_layout()
        
    fig2, ax2 = plt.subplots()
    fig2.set_size_inches(19.4, 10)
    
    avg_data["NN_overhead_1_train_3"] = avg_data["NN_overhead_1_train_3"] + ((61/(np.array(syncs)/1000))*8057152)
    
    for i in range(2):
        for j in range(2):
            ax2.plot(avg_data[f"{['NN', 'target'][i]}_overhead_{(0, 1)[j]}_train_3"]/(512*8*(avg_data[f"{['NN', 'target'][i]}_total_rew_{(0, 1)[j]}_train_3"]/avg_data[f"{['NN', 'target'][i]}_real_rew_{(0, 1)[j]}_train_3"])),
                     avg_data[f"{['NN', 'target'][i]}_real_rew_{(0, 1)[j]}_tests_3"], 
                     label=f"{model_names[i]}_{(0, 1)[j]}", linestyle=line_styles[i], marker="o")
    fig2.legend(loc=2)
    fig2.tight_layout()
    ax2.set_xlabel(f"Signaling Overhead (%) ")
    ax2.set_ylabel(f"Average Cost per packet")
    # plt.savefig(f"{folder_path}\\{folder_name}\\{stat_file_names[kkk]}_sync_step_variation_mse_offband.png")
    plt.show()