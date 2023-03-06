import os
import pickle
import numpy as np

path = '0'
obj = []
with open(path, 'rb') as file:
        try:
            while True:
                obj = pickle.load(file)
                #print(obj)
                np.savetxt(path+"_txt.txt", np.array(obj), fmt="%s")
        except EOFError:
            pass