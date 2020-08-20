import os
import re
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
import seaborn as sns
from scipy import stats

dataset = "multi"
#domain_list = ["real","sketch","clipart","painting"]
domain_list = ["painting"]
class_and_num_list_sorted = []
threshold = 300
master_list = []

for domain in domain_list:
    class_list = "lists/" + dataset + "-class-list.txt"
    # Making the list of classes
    f = open(class_list)
    my_file = f.read().split("\n")
    f.close()
    class_list = []

    for item in my_file:
        item = str(item)
        m = re.search('/(.+?)/',item)
        if m:
            found = m.group(1)
        class_list.append(found)

    del[class_list[len(class_list)-1]] # the list of classes for the given dataset is made here

    data_path =  os.path.join("../data",dataset,domain)

    # List which stored number of examples per class
    class_num_list = []
    i = 0
    for class_ in class_list:
        num_class = len(os.listdir(os.path.join(data_path,class_)))
        class_num_list.append(num_class)

    class_num_list = np.array(class_num_list)
    idx_list = (class_num_list>=threshold).astype(int)
    non_zero = idx_list.nonzero()
    non_zero = non_zero[0]
    class_list = [class_list[i] for i in non_zero]
    master_list.append(class_list)

class_list.sort()
print(len(class_list))
#list3 = list(set(master_list[0]) & set(master_list[1]))
# Convert list to set
set_list = []
for item in master_list:
    item = set(item)
    set_list.append(item)

#u = set.intersection(set_list[0],set_list[1],set_list[2],set_list[3])
#u = set.intersection(set_list[0],set_list[1])
#u = list(u)
#print(u)
#print(len(u))

"""
with open('lists/class_list.txt', 'w') as f:
    for item in u:
        f.write("%s\n" % item)

"""



"""
for i in range(len(domain_list)):
    print(domain_list[i])
    print("\n")
    for item1, item2 in class_and_num_list_sorted[i]:
        print(item1,item2)
    print("_______________________________________________________________________________________")
"""
