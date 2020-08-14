import os
import re
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
import seaborn as sns
from scipy import stats

dataset = "multi"
domain = "clipart"
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

#print(class_num_list)
#print("Mean is: ", sum(class_num_list)/len(class_num_list))
# Sorting the classes according to the number of samples
s = [x for _,x in sorted(zip(class_num_list,class_list))]
print(s)
class_num_list.sort()
print(class_num_list)

kde = stats.gaussian_kde(np.array(class_num_list))
xx = np.linspace(0, max(class_num_list), 10000)
fig, ax = plt.subplots(figsize=(8,6))


ax.hist(np.array(class_num_list), density=False,bins = np.arange(0,max(class_num_list),50), edgecolor = "black", color = "peachpuff")
ax.plot(xx,kde(xx)*6285, color = "orange")

#ax.hist(np.array(class_num_list), density=False,bins = np.arange(0,max(class_num_list),50))
#ax.plot(xx,kde(xx))


plt.xlabel("Number of Examples", fontsize = 20)
plt.ylabel("Number of Classes", fontsize = 20)
plt.title("\"%s\" Domain Distribution"%domain, fontsize = 20)
plt.show()
