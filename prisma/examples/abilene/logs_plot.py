import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

## define the parameters
link_cap = 500000 #bps
hop_delay = 0.001 #s
pkt_size = 542 #bytes

nb_nodes = 4
exp=f"dqn_buffer_{nb_nodes}n_moving_avg_variation"
for session in os.listdir(f"examples/abilene/{exp}"):
    if session in ("opt", "sp", "saved_models"):
        continue
    # session=f"dqn_buffer_10_explo_10_fr10"
    nodes = list(range(nb_nodes))#[0,1,2,3,4]
    interfaces = list(range(nb_nodes-2))
    logs=["ground_truth_delay", "delay", "rew"]
    for node in nodes:
        for interface in interfaces:
            fig, ax = plt.subplots()
            for log in logs:
            
                filename = f"examples/abilene/{exp}/{session}/{log}_{node}_{interface}.txt"
                if log == "ground_truth_delay":
                    delim = ' '
                else:
                    delim = '  '
                Data = np.genfromtxt(filename, dtype=float, delimiter=delim)

                time = Data[:,0]
                value = Data[:,1]
                
                if log in ("rew", "ground_truth_delay"):
                    value *= 1000
                    

                    
                ax.plot(time, value, label=log, alpha=0.7)

            #filename = f"{exp}/{session}/tsent_{node}_{interface}.txt"
            #Data = np.genfromtxt(filename, dtype=float, delimiter="  ")
            #time = Data[:]
            #value = np.zeros(time.shape)
            #ax.plot(time, value, 'o', label="Tsent")
            ax.set_xlim((0, 100))
            ax.legend()
            if exp not in os.listdir("/home/redha/PRISMA/prisma/examples/abilene/plots/"):
                os.mkdir(f"/home/redha/PRISMA/prisma/examples/abilene/plots/{exp}")
            if session not in os.listdir(f"/home/redha/PRISMA/prisma/examples/abilene/plots/{exp}"):
                os.mkdir(f"/home/redha/PRISMA/prisma/examples/abilene/plots/{exp}/{session}")
            
            plt.savefig(f"/home/redha/PRISMA/prisma/examples/abilene/plots/{exp}/{session}/{node}_{interface}.png")
            # plt.show()
            # print()