import os
import sys


# Real to Painting

# Function to get the number of examples per class given the number of classes and 
def get_img_num_per_cls(total_examples, cls_num, imb_type, imb_factor):
    img_max = total_examples / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(25):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(20):
            img_num_per_cls.append(int(img_max * imb_factor))

    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls


# Reading the file to be imbalanced - handle everything as lists
f = open("../data/txt/multi_1/labeled_source_images_real.txt","r")
f = [line for line in f]
f = [line.replace("\n"," ") for line in f]

num_classes = 45
total_examples = 4500
# make this as a list of lists
# Initialize the list of lists
example_list = []
for i in range(num_classes):
    mini_list = []
    example_list.append(mini_list)


for class_ in range(num_classes):
    to_check_str = " "  + str(class_) + " "
    for line in f:
        if to_check_str in line:
            example_list[class_].append(line[:-1])


imbalanced_example_list = []
for i in range(num_classes):
    mini_list = []
    imbalanced_example_list.append(mini_list)


img_num_per_cls = get_img_num_per_cls(total_examples,num_classes,"exp",0.1)

for class_idx, img_num in enumerate(img_num_per_cls):
    imbalanced_example_list[class_idx] = example_list[class_idx][0:img_num]

f = open("labeled_source_images_real.txt","w")

# Writing imbalanced data list into a file 
for class_idx in range(num_classes):
    for example in imbalanced_example_list[class_idx]:
        f.write(example)
        f.write("\n")

print(img_num_per_cls)
print(len(img_num_per_cls))
