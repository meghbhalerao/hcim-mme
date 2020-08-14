import os
import re
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
import seaborn as sns
from scipy import stats

dataset = "multi"
domain_list = ["real","sketch","clipart","painting"]

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

    s = [x for _,x in sorted(zip(class_num_list,class_list))]
    print(s)
    class_num_list.sort()
    print(class_num_list)