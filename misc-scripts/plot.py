import numpy as np
import matplotlib.pyplot as plt
import re

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

print(class_list)
