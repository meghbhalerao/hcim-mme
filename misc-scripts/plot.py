import numpy as np
import matplotlib.pyplot as plt
import re
import os

# class list dirty strings
class_num_actual = "real_num_actual_pc.txt"
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
class_num_actual = np.array(class_num_actual)
class_num_pred = np.array(class_num_pred)


# sorting both the arrays accoring to the actual number of exmaples per class
print(abs(np.sum(class_num_pred-class_num_actual))/sum(class_num_actual))
arr1inds = class_num_actual.argsort()
class_num_actual = class_num_actual[arr1inds]
class_num_pred = class_num_pred[arr1inds]


plt.plot(class_num_pred, label = "predicted target distribution by class of target examples")
plt.plot(class_num_actual, label = "actual source distribution by class of target examples")
plt.legend()
plt.title("Distributions")
plt.xlabel('Class')
plt.ylabel('# examples in the class')
plt.show()
