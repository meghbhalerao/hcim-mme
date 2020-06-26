import numpy as np
import matplotlib.pyplot as plt
import re
import os

# class list dirty strings
class_num_actual = "sketch_num_actual_pc.txt"
class_num_pred = "sketch_num_pred_pc.txt"
# Making the list of classes
f_act = open(class_num_actual)
f_act = f_act.read().split("\n")
class_num_actual = []
for item in f_act:
    if item != '':
        class_num_actual.append(int(item))

# Making the list of classes
f_act = open(class_num_pred)
f_act = f_act.read().split("\n")
class_num_pred = []
for item in f_act:
    if item != '':
        class_num_pred.append(int(item))

plt.plot(class_num_pred, label = "predicted distribution by class of target examples")
plt.plot(class_num_actual, label = "actual distribution by class of target examples")
plt.legend()
plt.savefig("class_distribution.png")
plt.show()

