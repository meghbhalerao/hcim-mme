import numpy as np
cf = np.load("cf_target.npy")
class_num_list = np.sum(cf,axis=0)

f = open("file.txt", "w")
for val in class_num_list:
    f.write(str(int(val)) + '\n')
