import networkx as nx
import numpy as np
import os
import sys
sys.path.append('../')
from source.utils import load_model
import matplotlib.pyplot as plt
src = 8
all_dst = list(range(11))
all_dst.remove(src)
loaded_models = load_model("../examples/abilene/saved_models/abilene_test_tiago_v2/iteration1_episode1")
G=nx.DiGraph(nx.empty_graph())
G = nx.from_numpy_matrix(np.loadtxt(open("../examples/abilene/adjacency_matrix.txt")), parallel_edges=False, create_using=G)
for dst in all_dst:
    neighbors = list(G.neighbors(src))
    if(len(neighbors)==2):
        x = np.arange(0, 30, 1)
        y = np.arange(0, 30, 1)
        xx, yy = np.meshgrid(x, y)
        z = np.zeros((len(y), len(x)))
        for i, item_x in enumerate(x):
            for j, item_y in enumerate(y):
                z[i][j] = np.argmin(loaded_models[src](np.array([[dst, item_x, item_y]])))
                #print(z)
        #print(dst, sum(z), "#"*7)
        print(z)
        print(xx.shape, yy.shape, z.ndim)
        plt.figure()
        cs = plt.contourf(xx, yy, z, cmap='jet', levels=[-0.1, 0.9, 1.4])
        print(cs.collections)
        proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in cs.collections]
        plt.legend(proxy, [f"go to {neighbors[0]}", f"go to {neighbors[1]}"])
        plt.xlabel(f"node {neighbors[0]} buffer length")
        plt.ylabel(f"node {neighbors[1]} buffer length")
        plt.xticks(x)
        plt.yticks(y)
        plt.title(f"from {src} to {dst}")
        if(not os.path.exists("figures/")):
            os.mkdir("figures")
        plt.savefig(f"figures/from {src} to {dst}")
    else:

        if(not os.path.exists("logs/")):
            os.mkdir("logs")
        f = open(f"logs/from {src} to {dst}.txt", "w")
        src_labels = [neighbors]
        values=[0,5,10,15,20,25,30]
        print(f"from {src} to {dst}")
        f.write(f"from {src} to {dst}\n")

        print("node {:<2}| node {:<2}| node {:<2}| {:<1}".format(neighbors[0], neighbors[1], neighbors[2], "action"))
        f.write("node {:<2}| node {:<2}| node {:<2}| {:<1}\n".format(neighbors[0], neighbors[1], neighbors[2], "action"))
        
        print('-'*32)
        f.write('-'*32+'\n')
        for a in (values):
            for b in (values):
                for c in (values):
                    print("{:<7}| {:<7}| {:<7}| go to {:<7}".format(a, b, c,neighbors[np.argmin(loaded_models[src](np.array([[dst, a, b, c]])))]))
                    f.write("{:<7}| {:<7}| {:<7}| go to {:<7}\n".format(a, b, c,neighbors[np.argmin(loaded_models[src](np.array([[dst, a, b, c]])))]))
        print()
        print()
        print()
        f.close()