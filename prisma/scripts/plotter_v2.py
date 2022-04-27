import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


metrics = ['loss_rate' ,'delay', 'rw']
colors = ['blue', 'red', 'green', 'magenta', 'cyan']

for met in metrics:
    df = pd.read_csv("../res_new_"+met, header=None)
    df[0] = pd.Categorical(df[0], ["sp", "opt", "dqn_buffer"])
    df[1] = pd.Categorical(df[1], ["ideal", "NN", "target"])
    df = df.sort_values([0,1,2,3,4])
    print(df.to_string())
    df_as_np = df.to_numpy()

    d = {}

    for i in df_as_np:
        if not i[2] in d.keys():
            d[i[2]] = {}
        if not (i[0], i[1]) in d[i[2]].keys():
            d[i[2]][(i[0], i[1])] = {}
        if not i[3] in d[i[2]][(i[0], i[1])].keys():
            d[i[2]][(i[0], i[1])][i[3]] = []
        d[i[2]][(i[0], i[1])][i[3]].append(i[4])

    for i in d.keys():
        f = plt.figure()
        for indx, j in enumerate(d[i].keys()):
            print(list(d[i][j].keys()))
            print(list(d[i][j].values()))
            plt.plot(list(d[i][j].keys()), list(d[i][j].values()), label = str(j), marker='o', color=colors[indx])
        plt.legend()
        plt.title(met + " - Matrix" + str(i))
        plt.ylabel(met)
        plt.xlabel("Load charge")
        plt.savefig(f'{met} - Matrix {i}')
#fig, ax = plt.subplots()
#bp = []
#colors = ['tomato', 'skyblue', 'magenta', 'green', 'orange']
#
#for indx,j in enumerate(d.keys()):
#    pos = 0.3*(float(indx-len(d.keys()))/2.0)
#    #print(d[j].values())
#    #print(j, list(d[j].values()))
#    bp1 = draw_plot(list(d[j].values()), pos, colors[indx], "white")
#    bp.append(bp1)
##print(d[('sp','ideal')].keys())
#ax.set_xticks(ticks=range(len(list(d[('sp','ideal')].keys()))) ,labels=list(d[('sp','ideal')].keys()))
#plt.xticks(rotation=90)
#plt.xlabel(f"load")
#plt.ylabel("Loss rate")
#ax.legend([bp[0]["boxes"][0], bp[1]["boxes"][0], bp[2]["boxes"][0], bp[3]["boxes"][0], bp[4]["boxes"][0]], ['SP', 'Ideal', 'NN', 'target','optimal'], loc='upper left')
#plt.tight_layout()
#plt.savefig(f"compare_loss_rate.png")
##print(d[('sp','ideal')])
###-------------------------------------------------------------------------------------------------------------------##
#df = pd.read_csv("../res_delay", header=None)
#df_as_np = df.to_numpy()
#
#d = {}
#
#for i in df_as_np:
#    if not (i[0], i[1]) in d.keys():
#        d[(i[0], i[1])] = {}
#    if not i[3] in d[(i[0], i[1])].keys():
#        d[(i[0], i[1])][i[3]] = []
#    d[(i[0], i[1])][i[3]].append(i[4])
#
#
#
#fig, ax = plt.subplots()
#bp = []
#colors = ['tomato', 'skyblue', 'magenta', 'green', 'orange']
#for indx,j in enumerate(d.keys()):
#    pos = 0.2*(float(indx-len(d.keys()))/2.0)
#    #print(d[j].values())
#    bp1 = draw_plot(list(d[j].values()), pos, colors[indx], "white")
#    bp.append(bp1)
##print(d[('sp','ideal')].keys())
#ax.set_xticks(ticks=range(len(list(d[('sp','ideal')].keys()))) ,labels=list(d[('sp','ideal')].keys()))
#plt.xticks(rotation=90)
#plt.xlabel(f"load")
#plt.ylabel("Delay")
#ax.legend([bp[0]["boxes"][0], bp[1]["boxes"][0], bp[2]["boxes"][0], bp[3]["boxes"][0], bp[4]["boxes"][0]], ['SP', 'Ideal', 'NN', 'target','optimal'], loc='upper left')
#plt.tight_layout()
#plt.savefig(f"compare_delay.png")
##print(d[('sp','ideal')])
#
###-------------------------------------------------------------------------------------------------------------------------------------------------##
#df = pd.read_csv("../res_rw", header=None)
#df_as_np = df.to_numpy()
#
#d = {}
#
#for i in df_as_np:
#    if not (i[0], i[1]) in d.keys():
#        d[(i[0], i[1])] = {}
#    if not i[3] in d[(i[0], i[1])].keys():
#        d[(i[0], i[1])][i[3]] = []
#    d[(i[0], i[1])][i[3]].append(i[4])
#
#
#
#fig, ax = plt.subplots()
#bp = []
#colors = ['tomato', 'skyblue', 'magenta', 'green', 'orange']
#for indx,j in enumerate(d.keys()):
#    pos = 0.5*(float(indx-len(d.keys()))/2.0)
#    #print(d[j].values())
#    bp1 = draw_plot(list(d[j].values()), pos, colors[indx], "white")
#    bp.append(bp1)
##print(d[('sp','ideal')].keys())
#ax.set_xticks(ticks=range(len(list(d[('sp','ideal')].keys()))) ,labels=list(d[('sp','ideal')].keys()))
#plt.xticks(rotation=90)
#plt.xlabel(f"load")
#plt.ylabel("Reward")
#ax.legend([bp[0]["boxes"][0], bp[1]["boxes"][0], bp[2]["boxes"][0], bp[3]["boxes"][0], bp[4]["boxes"][0]], ['SP', 'Ideal', 'NN', 'target','optimal'], loc='upper left')
#plt.tight_layout()
#plt.savefig(f"compare_rw.png")
##print(d[('sp','ideal')])