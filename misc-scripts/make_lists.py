import os
dataset = "multi"
domain = "real"
with open(os.path.join("lists",dataset + "_balanced_class_list.txt")) as f:
    class_list = [line.rstrip() for line in f]

data_path = os.path.join("../data",dataset,domain,)