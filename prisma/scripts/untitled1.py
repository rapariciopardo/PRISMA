# -*- coding: utf-8 -*-
"""
Created on Mon May  2 14:43:32 2022

@author: Redha
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

plt.rcParams.update({'font.size': 22})

if __name__ == '__main__':
    ### define params
    stat_names = ["Loss", "Delay", "Theoretical Cost", "Avg Cost", "total cost"]
    stat_file_names = ["loss", "delay", "rew", "real_rew", "total_rew"]
    folder_path = "D:/NetSim_/NetSim/prisma/"
    
    kkk = 3
    """ 1 ==  sync step variation for ideal case 
        2 == signaling type comparaison for sync step 1s and 20% ratio 
        3 == ratio comparaison  for NN signaling 
    """
    folder_name =  f"tests_plots{kkk}"
    
    ### check if folder exists
    if folder_name not in os.listdir(folder_path):
        os.mkdir(folder_path + "\\" + folder_name)
        
    for stat_idx in range(len(stat_names)):
        fig = plt.figure()
        fig.set_size_inches(19.4, 10)
        #list_charges = [round(x, 4) for x in list(np.arange(0.4, 1.5, 0.1))]
        list_charges = [60, 70, 80, 90, 100, 110, 120, 130, 140]
        #sync_step = 15
        # names = ["giant_final_dqn_kernel_best_over_10eps", "giant_final_dqrc_best", "giant_final_q_routing_best",]
        
        if kkk ==2:
            names = [
                "dqn_buffer_ideal_sync1000ms",
                 "dqn_buffer_NN_sync-1000ms",
                 "dqn_buffer_target_sync1000ms",
                 "sp_ideal_sync1000ms",
                 "opt_ideal_sync1000ms",]
            color = [ "purple", "black", "green", "red","blue"]
            official_names = [ 
                                "dqn_buffer_ideal_sync 1s",
                                 "dqn_buffer_NN_sync 500ms ratio 20%",
                                 "dqn_buffer_target_sync 1s",
                                 "Shortest Path",
                                 "Optimal Solution"]
            increment = [-1.2, -0.6, 0, 0.6, 1.2]
        elif kkk==1:
            names = [
                    "dqn_buffer_ideal_sync500ms",
                      "dqn_buffer_ideal_sync1000ms",
                      "dqn_buffer_ideal_sync1500ms",
                      "dqn_buffer_ideal_sync2000ms",
                      "dqn_buffer_ideal_sync2500ms",
                      "dqn_buffer_ideal_sync3000ms",
                      "sp_ideal_sync1000ms"]
            official_names = [   "dqn_buffer_ideal_sync 500ms",
                              "dqn_buffer_ideal_sync 1s",
                              "dqn_buffer_ideal_sync 1.5s",
                              "dqn_buffer_ideal_sync 2s",
                              "dqn_buffer_ideal_sync 2.5s",
                              "dqn_buffer_ideal_sync 3s",
                              "Shortest Path"]
            color = [ "gray", "black", "green", "purple", "orange", "blue", "red"]
            increment = [-1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8]
        elif kkk==3:
            names = [
                    "dqn_buffer_ideal_sync1000ms",
                      "dqn_buffer_NN_fixed_sync-1000ms_ratio_5",
                      "dqn_buffer_NN_fixed_sync-1000ms_ratio_10",
                      "dqn_buffer_NN_fixed_sync-1000ms",
                      "dqn_buffer_target_sync1000ms",
                      "sp_ideal_sync1000ms",
                      "opt_ideal_sync1000ms"]
            official_names = [   "dqn_buffer_ideal_sync 1s",
                              "dqn_buffer_NN_sync 3s ratio 5%",
                              "dqn_buffer_NN_sync 1s ratio 10%",
                              "dqn_buffer_NN_sync 500ms ratio 20%",
                              "dqn_buffer_target_sync 1s reduced Replay buffer",
                              "Shortest Path",
                              "Optimal Solution"]
            color = [ "gray", "black", "green", "purple", "orange", "blue", "red"]
            increment = [-1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8]
        # names = ["optimal_solution_30",
        #         f"abilene_dqn_buffer_almost_final_env_no_signaling_sync_{sync_step}_charge_0.4_out_buff_30",
        #          f"abilene_dqn_buffer_almost_final_env_no_signaling_sync_{sync_step}_charge_0.4_out_buff_30",
        #          f"abilene_dqn_buffer_almost_final_env_no_signaling_sync_{sync_step}_charge_0.4_out_buff_30",
        #          f"abilene_dqn_buffer_almost_final_env_no_signaling_sync_{sync_step}_charge_0.4_out_buff_30",
        #          "abilene_sp_new_30"]
        # names = ["optimal_solution_30",
        #         f"abilene_dqn_buffer_almost_final_env_no_signaling_sp_init_sync_{sync_step}_charge_0.4_out_buff_30",
        #         f"abilene_dqn_buffer_almost_final_env_no_signaling_sp_init_sync_{sync_step}_charge_0.4_out_buff_30",
        #         f"abilene_dqn_buffer_almost_final_env_no_signaling_sp_init_sync_{sync_step}_charge_0.4_out_buff_30",
        #         f"abilene_dqn_buffer_almost_final_env_no_signaling_sp_init_sync_{sync_step}_charge_0.4_out_buff_30",
        #         f"abilene_dqn_buffer_almost_final_env_no_signaling_sp_init_sync_{sync_step}_charge_0.4_out_buff_30",
        #         "abilene_sp_new_30"]
        


        # official_names = ["Optimal Solution over traff mat", 
        #                   f"DQN buffer no signalisation sync timestep {sync_step} traf mat ", 
        #                   f"DQN buffer no signalisation sync timestep {sync_step} traf mat ", 
        #                   f"DQN buffer no signalisation sync timestep {sync_step} traf mat ", 
        #                   f"DQN buffer no signalisation sync timestep {sync_step} traf mat ",
        #                   "SP over traff mat"]
        # official_names = ["Optimal Solution over traff mat", 
        #                 f"DQN buffer no signalisation SP init sync timestep {sync_step} seed ", 
        #                 f"DQN buffer no signalisation SP init sync timestep {sync_step} seed ", 
        #                 f"DQN buffer no signalisation SP init sync timestep {sync_step} seed ", 
        #                 f"DQN buffer no signalisation SP init sync timestep {sync_step} seed ",
        #                 f"DQN buffer no signalisation SP init sync timestep {sync_step} seed ",
        #                 "SP over traff mat"]
        
        # 
        # increment = [-1.5, -0.9, -0.3, 0.3, 0.9, 1.5]
        #increment = [-0.9, -0.3, 0.3, 0.9]
        j = 0
        for i in range(len(names)):
            for charge_index, charge in enumerate(list_charges):
                print(stat_names[stat_idx], names[i], charge)
                for k in [0]:
                    if k == 0:
                        delay_temp = np.array(pd.read_csv(f"D:\\NetSim_\\NetSim\\prisma\\examples\\abilene\\tests_\\{names[i]}_load_{charge}.txt").values[:, 5 + stat_idx], dtype=float).reshape(1, -1)
                    else:
                        delay_temp = np.concatenate((delay_temp, np.array(pd.read_csv(f"D:\\NetSim_\\NetSim\\prisma\\examples\\abilene\\tests_\\{names[i]}_load_{charge}.txt").values[:, 5 + stat_idx], dtype=float).reshape(1, -1)))
                
                if charge_index == 0:
                    delay = delay_temp.reshape(1, -1)
                else:
                    delay = np.concatenate((delay, delay_temp.reshape(1, -1)))
            # import seaborn as sns
            # /home/redha/Desktop/results25_avril/dqn_kernel_no_init
            bpl = plt.boxplot(x=np.transpose(delay[:, :20]), sym='', widths=0.5, positions=np.array(range(len(list_charges)))*len(names)+increment[j])
            # if i not in (0, 6):
            #     seed = i -1 
            #     inc = i * 100
            #     # traf_mat_idx = seed * 5 
            #     # bpl = plt.boxplot(x=np.transpose(delay[:,np.arange(traf_mat_idx, traf_mat_idx+5)]), sym='', widths=0.5,positions=np.array(range(len(list_charges)))*len(names)+increment[j])
            #     bpl = plt.boxplot(x=np.transpose(delay[:,np.arange(int(seed) ,20,5)]),showfliers =True, sym='', widths=0.5,positions=np.array(range(len(list_charges)))*len(names)+increment[j])
            #     print(np.arange(int(seed) ,20,5)) 
            #     # print(np.arange(traf_mat_idx, traf_mat_idx+5))
            # else: 
            #     seed = 0
            #     bpl = plt.boxplot(x=np.transpose(delay[:,np.arange(int(seed) ,20,5)]),showfliers =True, sym='', widths=0.5,positions=np.array(range(len(list_charges)))*len(names)+increment[j])
            #     inc = ""
                
            set_box_color(bpl, color[i]) 
            np.mean(delay, axis=1)
            plt.plot([],c=color[i], label=f"{official_names[i]}")
            # plt.plot([],c=color[i], label=f"{official_names[i]} {inc}")
            j += 1
        plt.xticks(range(0, len(list_charges) * (len(names)), len(names)), np.array((np.array(list_charges)), dtype=int))
        plt.legend(loc=2)
        plt.ylabel(f"{stat_names[stat_idx]}")
        plt.xlabel(f"Load charge")
        plt.tight_layout()
        plt.savefig(f"{folder_path}\\{folder_name}\\{stat_file_names[stat_idx]}.png")
    plt.show()