import os
import sys

dataset = "office"
domain = "amazon"
imbalance_type = "step"
imbalance_factor = 0.5

if dataset == "office":
    data_path = "../data/%s/%s/images/"%(dataset,domain)

num_examples = []
for class_ in os.listdir(data_path):
    num_examples.append(len(os.listdir(os.path.join(data_path,class_))))

num_examples.sort(reverse = True)
num_class = len(num_examples)
total_examples = sum(num_examples)

def get_img_num_per_cls(cls_num, imb_type, imb_factor):
    img_max = total_examples / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls


# Counting the number of examples in the list
f = open("../data/txt/%s/labeled_source_images_%s.txt"%(dataset,domain),"r")
i = 0
for line in f:
    i = i + 1

if i == total_examples:
    print("All good")
else:
    print("exiting program because of example mismatch")
    sys.exit()

path_mkdir = "../data/txt/%s_%s_%s"%(dataset,imbalance_type,str(imbalance_factor))
if not os.path.exists(path_mkdir):
    os.mkdir(path_mkdir)

imbalanced_class_list = get_img_num_per_cls(num_class, imbalance_type, imbalance_factor)

# Below is the program to 





