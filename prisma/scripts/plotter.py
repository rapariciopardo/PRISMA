import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def draw_plot(data, offset,edge_color, fill_color):
    #print(data)
    #print(offset)
    pos = np.arange(np.array(data).shape[0])*2+offset 
    #print(pos)
    bp = ax.boxplot(data, positions= pos, widths=0.2, patch_artist=True, manage_ticks=True)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)
    return bp

metrics = ['loss_rate', 'delay', 'rw']
for met in metrics:

    df = pd.read_csv("../res_new_"+met, header=None)
    df[0] = pd.Categorical(df[0], ["sp", "opt", "dqn_buffer"])
    df[1] = pd.Categorical(df[1], ["ideal", "NN", "target"])
    df = df.sort_values([0,1,2,3,4])
    df_as_np = df.to_numpy()

    d = {}

    for i in df_as_np:
        if not (i[0], i[1]) in d.keys():
            d[(i[0], i[1])] = {}
        if not i[3] in d[(i[0], i[1])].keys():
            d[(i[0], i[1])][i[3]] = []
        d[(i[0], i[1])][i[3]].append(i[4])



    fig, ax = plt.subplots()
    bp = []
    colors = ['tomato', 'skyblue', 'magenta', 'green', 'orange']

    for indx,j in enumerate(d.keys()):
        pos = 0.3*(float(indx-len(d.keys()))/2.0)
        #print(d[j].values())
        #print(j, list(d[j].values()))
        bp1 = draw_plot(list(d[j].values()), pos, colors[indx], "white")
        bp.append(bp1)
    #print(d[('sp','ideal')].keys())
    ax.set_xticks(ticks=range(len(list(d[('sp','ideal')].keys()))) ,labels=list(d[('sp','ideal')].keys()))
    plt.title(met)
    plt.xticks(rotation=90)
    plt.xlabel(f"load")
    plt.ylabel(met)
    #ax.legend([bp[0]["boxes"][0], bp[1]["boxes"][0], bp[2]["boxes"][0], bp[3]["boxes"][0], bp[4]["boxes"][0]], ['SP', 'Optimal', 'DQN-Ideal', 'DQN-NN','DQN-TARGET'], loc='upper left')
    #print(len(bp))
    ax.legend([bp[i]["boxes"][0] for i in range(len(bp))], ['SP', 'Optimal', 'DQN-Ideal', 'DQN-NN', 'DQN-Target'][:len(bp)], loc='upper left')

    plt.tight_layout()
    plt.savefig(f"compare_{met}.png")
#print(d[('sp','ideal')])
##-------------------------------------------------------------------------------------------------------------------##
