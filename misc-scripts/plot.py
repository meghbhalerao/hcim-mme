import numpy as np
import matplotlib.pyplot as plt
import re
import os

# class list dirty strings
class_list = "lists/multi-class-list.txt"

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

del[class_list[len(class_list)-1]]

# roor dir of the target examples
root = "../data/multi/sketch/"
class_num_list = []

for class_ in class_list:
    class_dir = root + class_
    length = len(os.listdir(class_dir))
    class_num_list.append(length)


f = open("file.txt", "w")
for val in class_num_list:
    f.write(str(val) + '\n')

