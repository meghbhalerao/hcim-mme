import numpy as np
import matplotlib.pyplot as plt
import re
import os

# class list dirty strings
class_list = "multi-class-list.txt"

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

f = open("classes.txt", "w")
for val in class_list:
    f.write(str(val) + '\n')

# roor dir of the source or target examples
root = "../../data/multi/painting/"
class_num_list = []

for class_ in class_list:
    class_dir = root + class_
    length = len(os.listdir(class_dir))
    class_num_list.append(length)


f = open("painting_class_num_list.txt", "w")
for val in class_num_list:
    f.write(str(val) + '\n')

