import numpy as np
import matplotlib.pyplot as plt
import re
import os

# class list dirty strings
class_num_actual_source = "real_num_actual_pc.txt"
class_num_actual_target = "sketch_num_actual_pc.txt"
class_num_pred = "sketch_num_pred_pc.txt"


# Making the list of classes and their number of samples
f_act = open(class_num_actual_source)
f_act = f_act.read().split("\n")
class_num_actual_source = []
for item in f_act:
    if item != '':
        class_num_actual_source.append(int(item))


# Making the list of classes
f_act = open(class_num_pred)
f_act = f_act.read().split("\n")
class_num_pred = []
for item in f_act:
    if item != '':
        class_num_pred.append(int(item))


# Making the list of classes
f_act = open(class_num_actual_target)
f_act = f_act.read().split("\n")
class_num_actual_target = []
for item in f_act:
    if item != '':
        class_num_actual_target.append(int(item))


class_num_actual_source = np.array(class_num_actual_source)
class_num_pred = np.array(class_num_pred)
class_num_actual_target = np.array(class_num_actual_target)

# sorting both the arrays accoring to the actual number of exmaples per class
#print(np.sum(abs(class_num_pred-class_num_actual))/sum(class_num_pred))

arr1inds = class_num_actual_target.argsort()
class_num_actual_target = class_num_actual_target[arr1inds]
class_num_pred = class_num_pred[arr1inds]
class_num_actual_source = class_num_actual_source[arr1inds]

plt.plot(class_num_pred, label = "predicted class target distribution")
plt.plot(class_num_actual_source, label = "actual class source distribution")
plt.plot(class_num_actual_target, label="actual class target distribution")
plt.legend()
plt.title("Distributions")
plt.xlabel('Class')
plt.ylabel('# examples in the class')
plt.show()
