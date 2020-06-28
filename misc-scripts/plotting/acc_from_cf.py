import numpy as np

# Making the list of classes
class_num_actual = "sketch_num_actual_pc.txt"
f_act = open(class_num_actual)
f_act = f_act.read().split("\n")
class_num_actual = []
for item in f_act:
    if item != '':
        class_num_actual.append(int(item))

class_num_actual = np.array(class_num_actual)

cf = np.load("cf_target.npy")
sum_row = np.sum(cf,axis=0)
i=0
acc_list = []
summ = 0
for i in range(126):
    acc_list.append(cf[i,i]/sum_row[i])
    summ = summ + abs(class_num_actual[i] - sum_row[i])
    print(abs(class_num_actual[i] - sum_row[i]))
acc_list = np.array(acc_list)

print(np.mean(acc_list))
print(summ/np.sum(cf))
print(np.trace(cf)/np.sum(cf))    
