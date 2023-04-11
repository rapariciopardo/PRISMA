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
plt.rcParams.update({'font.size': 60})
plt.rcParams.update({'font.weight': 'bold'})
# plt.rcParams.update({'font.serif': ["Times"]})
if __name__ == '__main__':
    ### define params
    stat_names = ["Average Loss Rate Over Loads", 
                  "Average Delay Per Arrived Pkts",
                  "Theoretical Cost",
                  "Average Cost Per Packet",
                  "Total cost", 
                  "Signaling Overhead in Bytes"]
    stat_file_names = ["loss", "delay", "rew", "real_rew", "total_rew", "overhead"]
    folder_path = "D:/NetSim_/NetSim/prisma/"
    all_data = {}
    for signaling in (1,):
        kkk = 3
        train_test = "tests_3"
        """ 1 ==  sync step variation for ideal case 
            2 == signaling type comparaison for sync step 1s and 20% ratio 
            3 == ratio comparaison  for NN signaling 
        """
        model_names = ["DQN Routing - Model Sharing", "DQN Routing - Value Sharing", "Shortest Path Routing", "LP - Min Cost MCF"]
        folder_name =  f"final_results"
        line_styles = ["solid" , "solid", "dashed", "dashed"]
        colors = [ "purple", "green", "red","blue"]
        ### check if folder exists
        if folder_name not in os.listdir(folder_path):
            os.mkdir(folder_path + "\\" + folder_name)
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(19, 17)
        
        for xxx, signaling_type in enumerate(["NN", "target", "sp", "opt"]):
            # for idx_mat in ['0', '1','2', '3']:
        
                for stat_idx in range(len(stat_names)):
                    if stat_idx != kkk:
                        continue
                    ## stat over sync step
                    # fig1 = plt.figure()
                    # fig1.set_size_inches(19.4, 10)
                    sync_steps = []
                    sync_overhead = []
                    
                    
                    # fig = plt.figure()
                    # fig.set_size_inches(19.4, 10)
                    list_charges = [60, 70, 80, 90, 100, 110, 120, 130, 140]
                    # list_charges = [40]
            
            
                    syncs = np.arange(500, 5000, 500).tolist() + np.arange(5000, 10000, 1000).tolist()
                    # syncs = np.arange(500, 5000, 500).tolist() + np.arange(5000, 25000, 5000).tolist()
                    # syncs.remove(4500)
                    if signaling_type in ("ideal", "NN"):
                        names = [f"dqn_buffer_{signaling_type}_{signaling}_fixed_sync{xx}ms_ratio_10" for xx in syncs] 
                        official_names = [f"DQN Buffer {signaling_type} sync {xx}s" for xx in np.array(syncs)/1000]
                
                    elif signaling_type == "target":
                        names = [f"dqn_buffer_target_{signaling}_fixed_sync10000ms_ratio_10"] * len(syncs) 
                        official_names = [f"DQN Buffer {signaling_type}"] * len(syncs) 
                    elif signaling_type == "sp":
                        names = ["sp_ideal_0_fixed_sync2500ms_ratio_10"]* len(syncs)
                        official_names = ["Shortest Path"]* len(syncs)
                    else:
                        names = ["opt_ideal_0_fixed_sync2500ms_ratio_10"] * len(syncs) 
                        official_names = ["Optimal Solution"] * len(syncs)
                    j = 0
                    for i in range(len(names)):
                        for charge_index, charge in enumerate(list_charges):
                            # print(stat_names[stat_idx], names[i], charge)
                            temp_all = np.loadtxt(f"D:\\NetSim_\\NetSim\\prisma\\examples\\abilene\\tests_3\\{names[i]}_load_{charge}.txt", delimiter=',', dtype=object)
                            # print(temp)
                            for idx_mat in ['0', '1','2', '3']:
                            # if True:
                                temp = temp_all[np.all(temp_all[:, 2:4] == [idx_mat, '100'], axis=1)][0].reshape(1, -1)
                                print("done", names[i], charge, temp.shape)
                                if len(temp):                        
                                    # if len(temp.shape) == 1:
                                    #     delay_temp = np.array(temp[6 + stat_idx], dtype=float).reshape(1, -1)
                                    # else:
                                    delay_temp = np.array(temp[:, 6 + stat_idx], dtype=float).reshape(1, -1)
                            
                                    if charge_index == 0:
                                        delay = delay_temp.reshape(1, -1)
                                    else:
                                        delay = np.concatenate((delay, delay_temp.reshape(1, -1)))
                                else:
                                    raise()
                                # raise()
                        # set_box_color(bpl, color[i]) 
                        # np.mean(delay, axis=1)
                        # if i > len(names) - 3:
                        #     plt.plot(list_charges, np.transpose(delay[:, :20])[0], label=f"{official_names[i]}",linestyle="dashed")
                        # else:
                        #     plt.plot(list_charges, np.transpose(delay[:, :20])[0], label=f"{official_names[i]}")
                        
                        sync_steps.append(np.mean(delay, axis=0))
                        # plt.plot([],c=color[i], label=f"{official_names[i]} {inc}")
                        j += 1
                    # plt.xticks(range(0, len(list_charges) * (len(names)), len(names)), np.array((np.array(list_charges)), dtype=int))
                    print(stat_names[stat_idx], sync_steps)
                    # plt.legend(loc=2)
                    # plt.ylabel(f"{stat_names[stat_idx]}")
                    # plt.xlabel(f"Load charge")
                    # plt.tight_layout()
                
                    # plt.savefig(f"{folder_path}\\{folder_name}\\{stat_file_names[stat_idx]}.png")
        
                    ax1.plot(np.array(syncs)/1000, np.array(sync_steps)[:, 0],
                             label=f"{model_names[xxx]}",
                             color=colors[xxx],
                             linestyle="solid",
                             marker="o",
                             linewidth=7,
                             markersize=20)
                    all_data[f"{model_names[xxx]}_{signaling}_{kkk}_{train_test}"] = np.array(sync_steps)[:, 0]
                    if xxx == 0:
                        min_x_value = syncs[np.argmin(sync_steps)]/1000
                        
    
        
        #%%
        kkk= 3
        train_test = "train_3"
        # ax2 = ax1.twinx()
        for xxx, signaling_type in enumerate(["NN", "target"]):
            # for idx_mat in ['0', '1','2', '3']:
        
                for stat_idx in range(len(stat_names)):
                    if stat_idx != kkk:
                        continue
                    ## stat over sync step
                    # fig1 = plt.figure()
                    # fig1.set_size_inches(19.4, 10)
                    sync_steps = []
                    sync_overhead = []
                    
                    
                    # fig = plt.figure()
                    # fig.set_size_inches(19.4, 10)
                    # list_charges = [60, 70, 80, 90, 100, 110, 120, 130, 140]
                    list_charges = [40]
            
            
                    syncs = np.arange(500, 5000, 500).tolist() + np.arange(5000, 10000, 1000).tolist()
                    # syncs = np.arange(500, 5000, 500).tolist() + np.arange(5000, 25000, 5000).tolist()
                    # syncs.remove(4500)
                    if signaling_type in ("ideal", "NN"):
                        names = [f"dqn_buffer_{signaling_type}_{signaling}_fixed_sync{xx}ms_ratio_20" for xx in syncs] 
                        official_names = [f"DQN Buffer {signaling_type} sync {xx}s" for xx in np.array(syncs)/1000]
                
                    elif signaling_type == "target":
                        names = [f"dqn_buffer_target_{signaling}_fixed_sync10000ms_ratio_20"] * len(syncs) 
                        official_names = [f"DQN Buffer {signaling_type}"] * len(syncs) 
                    elif signaling_type == "sp":
                        names = ["sp_ideal_0_fixed_sync2500ms_ratio_20"]* len(syncs)
                        official_names = ["Shortest Path"]* len(syncs)
                    else:
                        names = ["opt_ideal_0_fixed_sync2500ms_ratio_20"] * len(syncs) 
                        official_names = ["Optimal Solution"] * len(syncs)
                    j = 0
                    for i in range(len(names)):
                        for charge_index, charge in enumerate(list_charges):
                            # print(stat_names[stat_idx], names[i], charge)
                            temp_all = np.loadtxt(f"D:\\NetSim_\\NetSim\\prisma\\examples\\abilene\\train_3\\{names[i]}_load_{charge}.txt", delimiter=',', dtype=object)
                            # print(temp)
                            for idx_mat in ['0', '1','2', '3']:
                            # if True:
                                temp = temp_all[np.all(temp_all[:, 2:4] == [idx_mat, '100'], axis=1)][0].reshape(1, -1)
                                print("done", names[i], charge, temp.shape)
                                if len(temp):                        
                                    # if len(temp.shape) == 1:
                                    #     delay_temp = np.array(temp[6 + stat_idx], dtype=float).reshape(1, -1)
                                    # else:
                                    delay_temp = np.array(temp[:, 6 + stat_idx], dtype=float).reshape(1, -1)
                            
                                    if charge_index == 0:
                                        delay = delay_temp.reshape(1, -1)
                                    else:
                                        delay = np.concatenate((delay, delay_temp.reshape(1, -1)))
                                else:
                                    raise()
                                # raise()
                        # set_box_color(bpl, color[i]) 
                        # np.mean(delay, axis=1)
                        # if i > len(names) - 3:
                        #     plt.plot(list_charges, np.transpose(delay[:, :20])[0], label=f"{official_names[i]}",linestyle="dashed")
                        # else:
                        #     plt.plot(list_charges, np.transpose(delay[:, :20])[0], label=f"{official_names[i]}")
                        
                        sync_steps.append(np.mean(delay, axis=0))
                        # plt.plot([],c=color[i], label=f"{official_names[i]} {inc}")
                        j += 1
                    # plt.xticks(range(0, len(list_charges) * (len(names)), len(names)), np.array((np.array(list_charges)), dtype=int))
                    print(stat_names[stat_idx], sync_steps)
                    # plt.legend(loc=2)
                    # plt.ylabel(f"{stat_names[stat_idx]}")
                    # plt.xlabel(f"Load charge")
                    # plt.tight_layout()
                
                    # plt.savefig(f"{folder_path}\\{folder_name}\\{stat_file_names[stat_idx]}.png")
                    # plt.savefig(f"{folder_path}\\{folder_name}\\{stat_file_names[stat_idx]}.eps")
                    # ax2.plot(np.array(syncs)/1000, np.array(sync_steps)[:, 0], label=f"{model_names[xxx]} - Overhead", color=colors[xxx], linestyle="dashed", marker="o")
                    all_data[f"{model_names[xxx]}_{signaling}_{kkk}_{train_test}"] = np.array(sync_steps)[:, 0]
                    if xxx == 0:
                        min_x_value = syncs[np.argmin(sync_steps)]/1000
        #%%
        kkk= 4
        train_test = "train_3"
        # ax2 = ax1.twinx()
        for xxx, signaling_type in enumerate(["NN", "target"]):
            # for idx_mat in ['0', '1','2', '3']:
        
                for stat_idx in range(len(stat_names)):
                    if stat_idx != kkk:
                        continue
                    ## stat over sync step
                    # fig1 = plt.figure()
                    # fig1.set_size_inches(19.4, 10)
                    sync_steps = []
                    sync_overhead = []
                    
                    
                    # fig = plt.figure()
                    # fig.set_size_inches(19.4, 10)
                    # list_charges = [60, 70, 80, 90, 100, 110, 120, 130, 140]
                    list_charges = [40]
            
            
                    syncs = np.arange(500, 5000, 500).tolist() + np.arange(5000, 10000, 1000).tolist()
                    # syncs = np.arange(500, 5000, 500).tolist() + np.arange(5000, 25000, 5000).tolist()
                    # syncs.remove(4500)
                    if signaling_type in ("ideal", "NN"):
                        names = [f"dqn_buffer_{signaling_type}_{signaling}_fixed_sync{xx}ms_ratio_20" for xx in syncs] 
                        official_names = [f"DQN Buffer {signaling_type} sync {xx}s" for xx in np.array(syncs)/1000]
                
                    elif signaling_type == "target":
                        names = [f"dqn_buffer_target_{signaling}_fixed_sync10000ms_ratio_20"] * len(syncs) 
                        official_names = [f"DQN Buffer {signaling_type}"] * len(syncs) 
                    elif signaling_type == "sp":
                        names = ["sp_ideal_0_fixed_sync2500ms_ratio_20"]* len(syncs)
                        official_names = ["Shortest Path"]* len(syncs)
                    else:
                        names = ["opt_ideal_0_fixed_sync2500ms_ratio_20"] * len(syncs) 
                        official_names = ["Optimal Solution"] * len(syncs)
                    j = 0
                    for i in range(len(names)):
                        for charge_index, charge in enumerate(list_charges):
                            # print(stat_names[stat_idx], names[i], charge)
                            temp_all = np.loadtxt(f"D:\\NetSim_\\NetSim\\prisma\\examples\\abilene\\train_3\\{names[i]}_load_{charge}.txt", delimiter=',', dtype=object)
                            # print(temp)
                            for idx_mat in ['0', '1','2', '3']:
                            # if True:
                                temp = temp_all[np.all(temp_all[:, 2:4] == [idx_mat, '100'], axis=1)][0].reshape(1, -1)
                                print("done", names[i], charge, temp.shape)
                                if len(temp):                        
                                    # if len(temp.shape) == 1:
                                    #     delay_temp = np.array(temp[6 + stat_idx], dtype=float).reshape(1, -1)
                                    # else:
                                    delay_temp = np.array(temp[:, 6 + stat_idx], dtype=float).reshape(1, -1)
                            
                                    if charge_index == 0:
                                        delay = delay_temp.reshape(1, -1)
                                    else:
                                        delay = np.concatenate((delay, delay_temp.reshape(1, -1)))
                                else:
                                    raise()
                                # raise()
                        # set_box_color(bpl, color[i]) 
                        # np.mean(delay, axis=1)
                        # if i > len(names) - 3:
                        #     plt.plot(list_charges, np.transpose(delay[:, :20])[0], label=f"{official_names[i]}",linestyle="dashed")
                        # else:
                        #     plt.plot(list_charges, np.transpose(delay[:, :20])[0], label=f"{official_names[i]}")
                        
                        sync_steps.append(np.mean(delay, axis=0))
                        # plt.plot([],c=color[i], label=f"{official_names[i]} {inc}")
                        j += 1
                    # plt.xticks(range(0, len(list_charges) * (len(names)), len(names)), np.array((np.array(list_charges)), dtype=int))
                    print(stat_names[stat_idx], sync_steps)
                    # plt.legend(loc=2)
                    # plt.ylabel(f"{stat_names[stat_idx]}")
                    # plt.xlabel(f"Load charge")
                    # plt.tight_layout()
                
                    # plt.savefig(f"{folder_path}\\{folder_name}\\{stat_file_names[stat_idx]}.png")
        
                    # ax2.plot(np.array(syncs)/1000, np.array(sync_steps)[:, 0], label=f"{model_names[xxx]} - Overhead", color=colors[xxx], linestyle="dashed", marker="o")
                    all_data[f"{model_names[xxx]}_{signaling}_{kkk}_{train_test}"] = np.array(sync_steps)[:, 0]
                    if xxx == 0:
                        min_x_value = syncs[np.argmin(sync_steps)]/1000
        #%%
        kkk= 5
        train_test = "train_3"
        fig2, ax2 = plt.subplots()
        fig2.set_size_inches(19, 17)
        for xxx, signaling_type in enumerate(["NN", "target"]):
            # for idx_mat in ['0', '1','2', '3']:
        
                for stat_idx in range(len(stat_names)):
                    if stat_idx != kkk:
                        continue
                    ## stat over sync step
                    # fig1 = plt.figure()
                    # fig1.set_size_inches(19.4, 10)
                    sync_steps = []
                    sync_overhead = []
                    
                    
                    # fig = plt.figure()
                    # fig.set_size_inches(19.4, 10)
                    # list_charges = [60, 70, 80, 90, 100, 110, 120, 130, 140]
                    list_charges = [40]
            
            
                    syncs = np.arange(500, 5000, 500).tolist() + np.arange(5000, 10000, 1000).tolist()
                    # syncs = np.arange(500, 5000, 500).tolist() + np.arange(5000, 25000, 5000).tolist()
                    # syncs.remove(4500)
                    if signaling_type in ("ideal", "NN"):
                        names = [f"dqn_buffer_{signaling_type}_{signaling}_fixed_sync{xx}ms_ratio_20" for xx in syncs] 
                        official_names = [f"DQN Buffer {signaling_type} sync {xx}s" for xx in np.array(syncs)/1000]
                
                    elif signaling_type == "target":
                        names = [f"dqn_buffer_target_{signaling}_fixed_sync10000ms_ratio_20"] * len(syncs) 
                        official_names = [f"DQN Buffer {signaling_type}"] * len(syncs) 
                    elif signaling_type == "sp":
                        names = ["sp_ideal_0_fixed_sync2500ms_ratio_20"]* len(syncs)
                        official_names = ["Shortest Path"]* len(syncs)
                    else:
                        names = ["opt_ideal_0_fixed_sync2500ms_ratio_20"] * len(syncs) 
                        official_names = ["Optimal Solution"] * len(syncs)
                    j = 0
                    for i in range(len(names)):
                        for charge_index, charge in enumerate(list_charges):
                            # print(stat_names[stat_idx], names[i], charge)
                            temp_all = np.loadtxt(f"D:\\NetSim_\\NetSim\\prisma\\examples\\abilene\\train_3\\{names[i]}_load_{charge}.txt", delimiter=',', dtype=object)
                            # print(temp)
                            for idx_mat in ['0', '1','2', '3']:
                            # if True:
                                temp = temp_all[np.all(temp_all[:, 2:4] == [idx_mat, '100'], axis=1)][0].reshape(1, -1)
                                print("done", names[i], charge, temp.shape)
                                if len(temp):                        
                                    # if len(temp.shape) == 1:
                                    #     delay_temp = np.array(temp[6 + stat_idx], dtype=float).reshape(1, -1)
                                    # else:
                                    delay_temp = np.array(temp[:, 6 + stat_idx], dtype=float).reshape(1, -1)
                            
                                    if charge_index == 0:
                                        delay = delay_temp.reshape(1, -1)
                                    else:
                                        delay = np.concatenate((delay, delay_temp.reshape(1, -1)))
                                else:
                                    raise()
                                # raise()
                        # set_box_color(bpl, color[i]) 
                        # np.mean(delay, axis=1)
                        # if i > len(names) - 3:
                        #     plt.plot(list_charges, np.transpose(delay[:, :20])[0], label=f"{official_names[i]}",linestyle="dashed")
                        # else:
                        #     plt.plot(list_charges, np.transpose(delay[:, :20])[0], label=f"{official_names[i]}")
                        
                        sync_steps.append(np.mean(delay, axis=0))
                        # plt.plot([],c=color[i], label=f"{official_names[i]} {inc}")
                        j += 1
                    # plt.xticks(range(0, len(list_charges) * (len(names)), len(names)), np.array((np.array(list_charges)), dtype=int))
                    print(stat_names[stat_idx], sync_steps)
                    # plt.legend(loc=2)
                    # plt.ylabel(f"{stat_names[stat_idx]}")
                    # plt.xlabel(f"Load charge")
                    # plt.tight_layout()
                
                    # plt.savefig(f"{folder_path}\\{folder_name}\\{stat_file_names[stat_idx]}.png")
                    # raise()
                    if signaling_type == "target":
                        sync_steps = np.array(sync_steps) * (80/72)
                    sync_steps = np.array(sync_steps)
                    if signaling_type == "NN" and signaling ==1:
                        sync_steps[:,0]+= ((60/(np.array(syncs)/1000))*8057152)
                    sync_steps[:, 0]=np.array(sync_steps)[:, 0]/(all_data[f"{model_names[xxx]}_{signaling}_4_train_3"]/all_data[f"{model_names[xxx]}_{signaling}_3_train_3"]*512*8)
                    ax2.plot(np.array(syncs)/1000,
                             sync_steps[:, 0],
                             label=f"{model_names[xxx]}", color=colors[xxx],
                             linestyle="solid",
                             marker="o",
                             linewidth=7,
                             markersize=20)
                    all_data[f"{model_names[xxx]}_{signaling}_{kkk}_{train_test}"] = np.array(sync_steps)[:, 0]
                    if xxx == 0:
                        min_x_value = syncs[np.argmin(sync_steps)]/1000
                    
        # ax1.vlines(min_x_value, 0, 3.5, linestyles="dotted", label="Selected value")
        
        
        
        # ax2.set_yl1im(0, )
        ax1.set_ylabel("Average Cost Per Packets", fontweight="bold")
        # ax2.tick_params(axis='y')
        
        # ax2.set_ylim(0, 8)
        # ax1.set_ylabel(f"{stat_names[kkk]}", fontweight="bold")
        ax1.set_xlabel(r"Target Update Period U (s)", fontweight="bold")
    
        ax1.set_ylim(0, 2.1)
        ax1.set_xlim(0, 9.5) 
        fig1.tight_layout()
        ax1.set_xticks(np.arange(0, 9, 2).tolist() , np.arange(0, 9, 2).tolist())
        fig1.legend(fontsize=40, loc=(0.2, 0.66), prop={'weight':'normal'})
    
        ax2.set_xlim(0, 9.5)
        ax2.set_ylabel("Signaling Overhead Ratio Over Data (%)", fontweight="bold")
        ax2.set_xticks(np.arange(0, 9, 2).tolist() , np.arange(0, 9, 2).tolist())
        ax2.set_xlabel(r"Target Update Period U (s)", fontweight="bold")
        fig2.legend(fontsize=40, loc=(0.2, 0.79), prop={'weight':'normal'})
        fig2.tight_layout()
        fig1.savefig(f"{folder_path}\\{folder_name}\\cost_vs_sync_steps.pdf", format="pdf")
        fig2.savefig(f"{folder_path}\\{folder_name}\\overhead_vs_sync_steps.pdf", format="pdf")
        fig1.savefig(f"{folder_path}\\{folder_name}\\cost_vs_sync_steps.eps", format="eps")
        fig2.savefig(f"{folder_path}\\{folder_name}\\overhead_vs_sync_steps.eps", format="eps")
        
    fig3, ax3 = plt.subplots()
    fig3.set_size_inches(19, 17)
    # ax3.plot(all_data['DQN Routing - Model Sharing_0_5_train_3'],
    #          all_data['DQN Routing - Model Sharing_0_3_tests_3'],
    #          label=f"{model_names[0]} - Offband",
    #          color="purple",
    #          linestyle="solid",
    #          marker="o",
    #          linewidth=7,
    #          markersize=15)
    ax3.plot(all_data['DQN Routing - Model Sharing_1_5_train_3'],
             all_data['DQN Routing - Model Sharing_1_3_tests_3'],
             label=f"{model_names[0]}",
             color="purple",
             linestyle="solid",
             marker="o",
             linewidth=7,
             markersize=20)
    
    # min_idx = np.argmin(all_data['DQN Routing - Value Sharing_0_3_tests_3'])
    # ax3.scatter(all_data['DQN Routing - Value Sharing_0_5_train_3'][min_idx],
    #          min(all_data['DQN Routing - Value Sharing_0_3_tests_3']),
    #          label=f"{model_names[1]} - Offband",
    #          color="green",
    #          marker="x",
    #          linewidth=30)
    
    min_idx = np.argmin(all_data['DQN Routing - Value Sharing_1_3_tests_3'])
    ax3.scatter(all_data['DQN Routing - Value Sharing_1_5_train_3'][min_idx],
             min(all_data['DQN Routing - Value Sharing_1_3_tests_3']),
             label=f"{model_names[1]}",
             color="green",
             marker="x",
             linewidth=40)
    
    ax3.hlines(all_data["Shortest Path Routing_1_3_tests_3"],
               0 ,
               12,
               label=f"{model_names[2]}",
               color="red",
               linestyle="dashed",
               linewidth=7)
    ax3.hlines(all_data["LP - Min Cost MCF_1_3_tests_3"],
               0 ,
               12,
               label=f"{model_names[3]}",
               color="blue",
               linestyle="dashed",
               linewidth=7)
    ax3.set_ylim(0.19, 1.35)
    ax3.set_xlim(0, 6.5)
    ax3.vlines(all_data['DQN Routing - Value Sharing_1_5_train_3'][min_idx],
               0, 10, linestyle="dotted")    
    ax3.vlines(all_data['DQN Routing - Model Sharing_1_5_train_3'][1],
                              0, 10, linestyle="dotted")
    ax3.vlines(all_data['DQN Routing - Model Sharing_1_5_train_3'][9],
                              0, 10, linestyle="dotted")
    # ax3.hlines(all_data['DQN Routing - Model Sharing_1_3_tests_3'][9],
    #                           0, 10, linestyle="dotted")
    fig3.legend(fontsize=40, loc=(0.195, 0.625), prop={'weight':'normal'})
    ax3.set_xlabel("Signaling Overhead Ratio Over Data (%)", fontweight="bold")
    ax3.set_ylabel("Average Cost Per Packets", fontweight="bold")
    ax3.set_xticks(np.arange(0, 8, 2).tolist() + [all_data['DQN Routing - Model Sharing_1_5_train_3'][9], all_data['DQN Routing - Model Sharing_1_5_train_3'][1]],
                   np.arange(0, 8, 2).tolist() + [np.round(all_data['DQN Routing - Model Sharing_1_5_train_3'][9],1), np.round(all_data['DQN Routing - Model Sharing_1_5_train_3'][1],1)])
    
    fig3.savefig(f"{folder_path}\\{folder_name}\\cost_vs_overhead_inband.pdf", format="pdf")
    fig3.savefig(f"{folder_path}\\{folder_name}\\cost_vs_overhead_inband.eps", format="eps")
    plt.show()