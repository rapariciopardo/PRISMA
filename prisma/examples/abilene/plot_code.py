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
plt.rcParams.update({'font.size': 50})
plt.rcParams.update({'font.weight': 'bold'})
# plt.rcParams.update({'font.serif': ["Times"]})
if __name__ == '__main__':
    ### define params
    stat_names = ["Average Cost Per Packets",
                  "Avg e2e Delay", 
                  "Avg loss rate"]
    stat_file_names = ["loss", "delay", "rew", "real_rew", "total_rew", "overhead"]
    folder_path = "."
    
    kkk = 3
    plot = 2 # 0-sync     1-rb    2-ind
    """ 1 ==  sync step variation for ideal case 
        2 == signaling type comparaison for sync step 1s and 20% ratio 
        3 == ratio comparaison  for NN signaling 
    """
    dqn_name= "NN"
    model_names = ["OPT", "SP"]
    folder_name =  f"tests_sync_variation_mat0_rb_10k_plots"
    line_styles = ["solid" , "solid", "dashed", "dashed", "solid"]
    colors = [ "purple", "green", "red", "orange"]
    avg_data = {}
    avg_data_loads = {}
    test_folder = "_tests_overlay_5n"
    signaling_inband=1
    underlay=1
    list_charges = [60, 70,80,90, 100,110, 120]
    #list_charges = [140]

    if(plot==0): 
        syncs = [1000, 10000, 20000, 30000, 40000, 50000]
        rb=[10000] #np.arange(1000, 8000, 1000).tolist()
    if(plot==1):
        syncs = [1000]
        rb = [512, 1024, 2500, 5000, 10000]#, 15000]
    if(plot==2):
        rb_size_ind = 10000
        sync_ind = 1000
        syncs = [sync_ind]
        rb = [rb_size_ind]
    overlayPackets = [100]#,20,50,100]#[5,10,20,50,100,500]
    agents = ["opt"]
    loads_train = [60,15,80,90,100,80,80]
                
    lite = ("", "","","", "", "")
    ### check if folder exists
    if folder_name not in os.listdir(folder_path):
        os.mkdir(folder_path + "/" + folder_name)
    for signaling_inband in (1,):
        for kkk_idx, kkk in enumerate((12, 11, 7)):
            # fig1, ax1 = plt.subplots()
            # fig1.set_size_inches(19.4, 10)
            # [0.6428656100517803, 0.47653319732663824, 0.44059512730308725, 0.36639229472221074, 0.5688626373310568, ]
            for xxx, signaling_type in enumerate(agents):
                #if(xxx==1):
                #    continue
                # if kkk_idx==1 and xxx >1:
                    # continue
                for stat_idx in range(len(stat_names)):
                    if stat_idx != kkk_idx:
                        continue
                sync_steps = []
                sync_charge_steps = []
                    
                    
                if signaling_type in ("ideal", "NN"):
                    print("here")
                    names = [f"ter_1000_20k_prio_0_underlayTraff_{underlay}_dqn_buffer_NN_{signaling_inband}_fixed_rb_{zz}_sync{xx}ms_ratio_10_overlayPackets_{yy}_loadTrain_{loads_train[xxx]}" for xx in syncs for yy in overlayPackets for zz in rb] 
                    official_names = [f"DQN Buffer {signaling_type} sync {xx}s overlayPackets {yy}" for xx in np.array(syncs)/1000 for yy in overlayPackets] 
                
                elif signaling_type in ("proportional"):
                    names = [f"ter_1000_20k_tr_0_underlayTraff_{underlay}_dqn_buffer_NN_{signaling_inband}_fixed_rb_{zz}_sync{xx}ms_ratio_10_overlayPackets_{yy}_loadTrain_{aa}" for xx in syncs for yy in overlayPackets for zz in rb for aa in [60]] 
                    official_names = [f"DQN Buffer {signaling_type} sync {xx}s overlayPackets {yy} Timeout" for xx in np.array(syncs)/1000 for yy in overlayPackets]
                elif signaling_type in ("No-Back"):
                    names = [f"ter_1000_20k_tr_0_underlayTraff_0_dqn_buffer_NN_{signaling_inband}_fixed_rb_{zz}_sync{xx}ms_ratio_10_overlayPackets_{yy}_loadTrain_{aa}" for xx in syncs for yy in overlayPackets for zz in rb for aa in [60]] 
                    official_names = [f"DQN Buffer {signaling_type} sync {xx}s overlayPackets {yy} Timeout" for xx in np.array(syncs)/1000 for yy in overlayPackets]
                elif signaling_type in ("timeout"):
                    names = [f"prio_0_underlayTraff_{underlay}_dqn_buffer_NN_{signaling_inband}_fixed_rb_{zz}_sync{xx}ms_ratio_10_overlayPackets_{yy}_loadTrain_{aa}_Timeout" for xx in syncs for yy in overlayPackets for zz in rb for aa in [80]] 
                    official_names = [f"DQN Buffer {signaling_type} sync {xx}s overlayPackets {yy} Timeout" for xx in np.array(syncs)/1000 for yy in overlayPackets]
                elif signaling_type in ("target"):
                    print("here")
                    if(plot==0):
                        names = [f"prio_0_underlayTraff_{underlay}_dqn_buffer_target_{signaling_inband}_fixed_rb_2500_sync1000ms_ratio_10_overlayPackets_{yy}_loadTrain_40" for yy in overlayPackets] 
                    else:
                        names = [f"prio_0_underlayTraff_{underlay}_dqn_buffer_target_{signaling_inband}_fixed_rb_{zz}_sync{xx}ms_ratio_10_overlayPackets_{yy}_loadTrain_180" for xx in syncs for yy in overlayPackets for zz in rb] 
                    official_names = [f"DQN Buffer {signaling_type} sync {xx}s overlayPackets {yy}" for xx in np.array(syncs)/1000 for yy in overlayPackets]  
                elif signaling_type in ("prio"):
                    names = [f"prio_1_dqn_buffer_target_{signaling_inband}_fixed_rb_10000_sync{xx}ms_ratio_10_overlayPackets_{yy}" for xx in syncs for yy in overlayPackets] 
                    official_names = [f"DQN Buffer {signaling_type} sync {xx}s overlayPackets {yy}" for xx in np.array(syncs)/1000 for yy in overlayPackets]
                
                elif signaling_type == "sp":
                    names = ["ter_t_1000_20k_tr_0_underlayTraff_1_sp_ideal_1_fixed_rb_10000_sync1000ms_ratio_10_overlayPackets_100_loadTrain_60"]* len(syncs)
                    official_names = ["Shortest Path"]* len(syncs)
                elif signaling_type == "opt":
                    names = ["ter_t_1000_20k_tr_0_underlayTraff_1_opt_ideal_1_fixed_rb_10000_sync1000ms_ratio_10_overlayPackets_100_loadTrain_60"] * len(syncs) 
                    official_names = ["Optimal Solution"] * len(syncs)
                j = 0
                for i in range(len(names)):
                    for charge_index, charge in enumerate(list_charges):
                        #if (signaling_type == "sp" or signaling_type == "opt") and charge == 40:
                        #    continue
                        temp = np.loadtxt(f"{test_folder}/{names[i]}_load_{charge}.txt", delimiter=',', dtype=object)
                        print(names[i], temp)
                        if len(temp.shape) == 1:
                            delay_temp = np.array(temp[kkk], dtype=float).reshape(1, -1)
                        else:
                            delay_temp = np.array(temp[:, kkk], dtype=float).mean(axis=0).reshape(1, -1)
                
                        if charge_index == 0:
                            delay = delay_temp.reshape(1, -1)
                        else:
                            delay = np.concatenate((delay, delay_temp.reshape(1, -1)))
                        
                    sync_steps.append(np.mean(delay, axis=0))
                    if(plot==2):
                        print(np.array(delay), xxx, f"{signaling_type}{lite[xxx]}_{stat_names[kkk_idx]}_{signaling_inband}_{test_folder}_{loads_train[xxx]}")
                        avg_data_loads[f"{signaling_type}{lite[xxx]}_{stat_names[kkk_idx]}_{signaling_inband}_{test_folder}_{loads_train[xxx]}"] = np.array(delay)[:, 0]
                       
                    sync_charge_steps.append(delay)
    
                    j += 1
                print(stat_names[kkk_idx], sync_steps)
                    
                #best_sync = np.argmin(np.array(sync_steps), axis=0)
                #print(best_sync)
                #print(sync_charge_steps[best_sync[0]])
                #
                #avg_data_loads[f"{signaling_type}{lite[xxx]}_{stat_names[kkk_idx]}_{signaling_inband}_{test_folder}"] = np.array(sync_charge_steps[best_sync[0]])[:, 0]
                avg_data[f"{signaling_type}{lite[xxx]}_{stat_names[kkk_idx]}_{signaling_inband}_{test_folder}"] = np.array(sync_steps)[:, 0]
                        
                #ax1.plot(np.array(syncs)/1000, avg_data[f"{signaling_type}_{kkk}_{signaling_inband}"] , label=f"{model_names[xxx]}", color=colors[xxx], linestyle=line_styles[xxx], marker="o")
        
                
            
            if(plot==2):
                fig2, ax2 = plt.subplots()
                fig2.set_size_inches(19.4, 10)
                for idx_agent, agent in enumerate(agents):
                    print(agent, idx_agent)
                    ax2.plot(np.array(list_charges)/100,
                         avg_data_loads[f"{agent}_{stat_names[kkk_idx]}_1_{test_folder}_{loads_train[idx_agent]}"],
                          label=f"{model_names[idx_agent]}", linestyle=line_styles[0],
                          marker="o",
                          color=colors[idx_agent],
                          linewidth=7,
                          markersize=20)
                #ax2.plot(np.array(list_charges)/100,
                #     avg_data_loads[f"teste"],
                #      label=f"DQN - model Sharing - 180%", linestyle=line_styles[0],
                #      marker="o",
                #      color=colors[3],
                #      linewidth=7,
                #      markersize=20)
                #if(kkk_idx==0): ax2.set_ylim(0.0, 2.0)
                #if(kkk_idx==1): ax2.set_ylim(30, 500)
                #if(kkk_idx==2): ax2.set_ylim(0.0, 0.5)
                ax2.set_xlim(0.6, 1.4)
                ax2.set_xticks(np.array(list_charges)/100) #(np.arange(1, 9, 1, dtype =int), np.arange(1, 9, 1, dtype =int))
                fig2.legend(prop={'weight':'normal'}, loc='lower right')
                fig2.tight_layout()
                ax2.set_xlabel(f"Load charge ", fontweight="bold")
                ax2.set_ylabel(f"{stat_names[kkk_idx]}", fontweight="bold")
                plt.savefig(f"pictures/avg_{stat_names[kkk_idx]}_overlay_load_inband.png")
                plt.show()
    #fig2, ax2 = plt.subplots()
    #fig2.set_size_inches(19.4, 10)
    #ax2.plot( avg_data['NN_Singalling overhead_1__train_final']/avg_data["NN_data_1__train_final"],
    #         avg_data["NN_Average Cost Per Packets_1__tests_final"],
    #          label=f"nn", linestyle=line_styles[0], marker="o")
    
    # ax2.plot( avg_data['NN_lite_Singalling overhead_1__train_final']/avg_data["NN_lite_data_1__train_final"],
    #          avg_data["NN_lite_Average Cost Per Packets_1__tests_final"],
    #           label=f"nn lite", linestyle=line_styles[0], marker="o")
    
    #fig2.legend(loc=2, prop={'weight':'normal'})
    #fig2.tight_layout()
    #ax2.set_xlabel(f"Signaling Overhead (%) ", fontweight="bold")
    #ax2.set_ylabel(f"Average Cost per packet", fontweight="bold")
    
    ## plot cost vs sync steps
    if(plot!=2):
        fig2, ax2 = plt.subplots()
        fig2.set_size_inches(19.4, 10)
        if(plot==0):
            x_value = np.array(syncs)/1000
        if(plot==1):
            x_value = np.array(rb)
        for idx_agent, agent in enumerate(agents):
            if(agent == "sp" or agent=="opt"):
                ax2.hlines(avg_data[f"{agent}_Average Cost Per Packets_1_{test_folder}"],
                            x_value[0] ,
                            x_value[-1],
                            label=f"SP",
                            color=colors[idx_agent],
                            linestyle="dashed",
                            linewidth=7)
            else:
                if(plot==0 and agent=='target'):
                    ax2.hlines(avg_data[f"{agent}_Average Cost Per Packets_1_{test_folder}"],
                            x_value[0] ,
                            x_value[-1],
                            label=f"Target Value",
                            color=colors[idx_agent],
                            linestyle="dashed",
                            linewidth=7)
                else:
                    ax2.plot(x_value,
                     avg_data[f"{agent}_Average Cost Per Packets_1_{test_folder}"],
                      label=f"{model_names[idx_agent]}", linestyle=line_styles[0],
                      marker="o",
                      color=colors[idx_agent],
                      linewidth=7,
                      markersize=20)
        
        
        ax2.set_xticks(np.array(x_value), np.array(x_value))#(np.arange(1, 9, 1, dtype =int), np.arange(1, 9, 1, dtype =int))
        fig2.legend(prop={'weight':'normal'})
        fig2.tight_layout()
        if(plot==0):
            ax2.set_xlabel(f"Sync step ", fontweight="bold")
        if(plot==1):
            ax2.set_xlabel(f"Replay Memory Size: ", fontweight="bold")
        ax2.set_ylabel(f"Average Cost per packet", fontweight="bold")


        ## plot cost vs sync steps
        #fig2, ax2 = plt.subplots()
        #fig2.set_size_inches(19.4, 10)
        #ax2.plot(np.array(syncs)/1000,
        #         avg_data['NN_Singalling overhead_1__train_final']/avg_data["NN_data_1__train_final"],
        #          label=f"nn", linestyle=line_styles[0], marker="o")
        #
        ## ax2.plot( np.array(syncs)/1000,
        ##          avg_data['NN_lite_Singalling overhead_1__train_final']/avg_data["NN_data_1__train_final"],
        ##           label=f"nn lite", linestyle=line_styles[0], marker="o")
        #
        #fig2.legend(prop={'weight':'normal'})
        #fig2.tight_layout()
#   
        #ax2.set_xlabel(f"Sync steps " ,fontweight="bold")
        #ax2.set_ylabel(f"Signaling Overhead (%) ", fontweight="bold")
#   
#   
        #print(avg_data["target_Average Cost Per Packets_1__tests_final"][0])



        # avg_data["NN_overhead_1_train_3"] = avg_data["NN_overhead_1_train_3"] + ((61/(np.array(syncs)/1000))*8057152)

        # for i in range(2):
        #     for j in range(2):
        if(plot==0):
            name_file_save = "pictures/avg_cost_overlay_sync_step_variation_inband.png"
        if(plot==1):
            name_file_save = "pictures/avg_cost_overlay_replay_buffer_size_variation_inband.png"
        
        plt.savefig(name_file_save)
        plt.show()