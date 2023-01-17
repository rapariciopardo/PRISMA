import numpy as np
import matplotlib.pyplot as plt

exp="4nodes"
session="test_log_4n_exp_0_1"
nodes = [0,1,2,3]
interfaces = [0,1]
logs=["rew", "delay", "tavg"]
for node in nodes:
    for interface in interfaces:
        fig, ax = plt.subplots()
        for log in logs: 
        
            filename = f"{exp}/{session}/{log}_{node}_{interface}.txt"

            Data = np.genfromtxt(filename, dtype=float, delimiter="  ")

            time = Data[:,0]
            value = Data[:,1]
            if(log=="rew"):
                value *= 1000

            ax.plot(time, value, label=log)

        filename = f"{exp}/{session}/tsent_{node}_{interface}.txt"
        Data = np.genfromtxt(filename, dtype=float, delimiter="  ")
        time = Data[:]
        value = np.zeros(time.shape)
        ax.plot(time, value, 'o', label="Tsent")

        ax.legend()
        plt.savefig(f"{exp}/{session}/{node}_{interface}.png")
        plt.show()