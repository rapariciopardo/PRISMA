from cProfile import label
import numpy as np
import matplotlib.pyplot as plt 
import os

def draw_plot(data, offset,edge_color, fill_color):
    print(data)
    pos = np.arange(data.shape[0])+offset 
    bp = ax.boxplot(data, positions= pos, widths=0.3, patch_artist=True, manage_ticks=True)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)
    return bp

if(not os.path.exists("figures/")):
    os.mkdir("figures")
print("Read logs from DQN")
f = open("../logs/log_dict_test_abilene_test_tiago_v2_3000.txt", "r")
A = f.readlines()
d ={}
for a in A:
  src = int(a.split(' ')[0])
  dest = int(a.split(' ')[1])
  hops = a.split(' ')[2]
  nhops = int(a.split(' ')[3])
  #if(src==8 and dest==4):
  #  print(hops)
  if(src, dest) in d:
    d[(src,dest)].append(nhops)
  else:
    d[(src, dest)] = [nhops]

print("Read logs from SP")
f = open("../logs/log_dict_test_sp_3000.txt", "r")
A = f.readlines()
d_sp ={}
for a in A:
  a= a.replace(", ", ",")
  src = int(a.split(' ')[0])
  dest = int(a.split(' ')[1])
  hops = a.split(' ')[2]
  nhops = int(a.split(' ')[3])
  #if(src==8 and dest==4):
  #  print(hops)
  if(src, dest) in d_sp:
    d_sp[(src,dest)].append(nhops)
  else:
    d_sp[(src, dest)] = [nhops]

step = 5#int(len(d)/10)
for i in range(0, int(len(d)/10),step):
  fig, ax = plt.subplots()
  print(d)
  bp1 = draw_plot(np.array(list(d.values())[i:i+step]), -0.2, "tomato", "white")
  bp2 = draw_plot(np.array(list(d_sp.values())[i:i+step]), +0.2,"skyblue", "white")
  ax.set_xticks(ticks=range(len(list(d.keys())[i:i+step])) ,labels=list(d.keys())[i:i+step])
  plt.xticks(rotation=90)
  plt.xlabel(f"source - destination")
  plt.ylabel("number of HOPS")
  ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['DQN', 'SP'], loc='upper right')
  plt.tight_layout()
  plt.savefig(f"figures/boxplots_nb_hops_{i}.png")
  
  
  #fig, ax = plt.subplots(figsize=(30,30))
  #ax.boxplot(list(d.values())[i:i+step], color=)
  #ax.boxplot(list(d_sp.values())[i:i+step])
  #ax.set_xticklabels(list(d.keys())[i:i+step])
  #plt.xticks(rotation=90)
  #plt.xlabel(f"source - destination")
  #plt.ylabel("number of HOPS")
  #plt.tight_layout()
  #plt.show()