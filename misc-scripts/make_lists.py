import os
dataset = "multi"
domain = "clipart"
shot = 3

"""
Description: This is the code to make the balanced example list

with open(os.path.join("lists",dataset + "_balanced_class_list.txt")) as f:
    class_list = [line.rstrip() for line in f]

data_path = os.path.join("../data",dataset,domain)

f = open(os.path.join("lists","labeled_source_images_" + domain + ".txt"),"w")
for label,class_ in enumerate(class_list):
    images = os.listdir(os.path.join(data_path,str(class_)))
    images = images[0:100]
    for image in images:
        f.write(os.path.join(domain,str(class_),str(image)))
        f.write(" ")
        f.write(str(label))
        f.write("\n")
"""

f = open(os.path.join("../data/txt/",dataset + "_balanced","labeled_source_images_%s.txt"%(domain)),"r")
#f = open(os.path.join("../data/txt/",dataset + "_balanced","unlabeled_target_images_%s_%s.txt"%(domain,str(shot))),"r")
f_validation = open(os.path.join("../data/txt/",dataset + "_balanced","validation_target_images_%s_%s.txt"%(domain,str(shot))),"w")

i = 0
for line in f:
    if i%100 == 3 or i%100 == 4 or i%100 == 5:
        f_validation.write(line)
        print(line)
    i = i +1 

# deleting those lines from the file
f_validation.close()
f.close()

"""
f_target_labeled = open(os.path.join("../data/txt/",dataset + "_balanced","labeled_target_images_%s_%s.txt"%(domain,str(shot))),"r")
lines_to_delete = f_target_labeled.readlines()

f = open(os.path.join("../data/txt/",dataset + "_balanced","unlabeled_target_images_%s_%s.txt"%(domain,str(shot))),"r")
master_list = f.readlines()

# loop to delete from master list
for to_delete in lines_to_delete:
    master_list.remove(to_delete)


# Writing the master list into the new unlabeled target example list
f = open(os.path.join("../data/txt/",dataset + "_balanced","unlabeled_target_images_%s_%s.txt"%(domain,str(shot))),"w")
for item in master_list:
    f.write(item)

"""
