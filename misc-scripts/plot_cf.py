import numpy as np
import matplotlib.pyplot as plt
cf = np.load("cf_target.npy")
class_num_list = np.sum(cf,axis=0)

plt.plot(class_num_list)
plt.show()
f = open("file.txt", "w")
for val in class_num_list:
    f.write(str(int(val)) + '\n')
