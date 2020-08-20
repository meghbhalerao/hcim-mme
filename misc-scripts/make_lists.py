import os
dataset = "multi"
domain = "clipart"
shot = 3


# Description: This is the code to make the balanced example list

with open("lists/r2p_class_list.txt") as f:
    class_list = [line.rstrip() for line in f]

# Making the source example list - later this can be put in a file - initially manipulate everything as files - becomes very easier
class_list.sort()

source_path = "../data/multi/real/"
labeled_source_images_real = []
for label,class_ in enumerate(class_list):
    images = os.listdir(os.path.join(source_path,str(class_)))
    images = images[0:100]
    labeled_source_images_real.append(images)

f = open("labeled_source_images_real.txt","w")
for idx, class_ in enumerate(labeled_source_images_real):
    for img in labeled_source_images_real[idx]:
        f.write("real/" + class_list[idx] + "/" + img + " " + str(idx) + "\n")


# Make the list for 300 target examples
target_path = "../data/multi/painting/"
labeled_target_images_painting = []

for label,class_ in enumerate(class_list):
    images = os.listdir(os.path.join(target_path,str(class_)))
    images = images[0:300]
    labeled_target_images_painting.append(images)
print(len(labeled_target_images_painting))

# Making the 1 shot and 3 shot image lists
unlabeled_target_images_painting_1 =[]
unlabeled_target_images_painting_3 = []

labeled_target_images_painting_1 =[]
labeled_target_images_painting_3 = []

validation_target_images_painting_3 = []


final_test_target = []


# Labelled target lists one shot and three shot make 
f = open("labeled_target_images_painting_1.txt","w")
for idx, class_ in enumerate(labeled_target_images_painting):
    line = str("painting/" + class_list[idx] + "/" + labeled_target_images_painting[idx][0]) + " " + str(idx) + "\n"
    f.write(line)


f = open("labeled_target_images_painting_3.txt","w")
for idx, class_ in enumerate(labeled_target_images_painting):
    line0 = str("painting/" + class_list[idx] + "/" + labeled_target_images_painting[idx][0]) + " " + str(idx) + "\n"
    line1 = str("painting/" + class_list[idx] + "/" + labeled_target_images_painting[idx][1]) + " " + str(idx) + "\n"
    line2 = str("painting/" + class_list[idx] + "/" + labeled_target_images_painting[idx][2]) + " " + str(idx) + "\n"
    f.write(line0)
    f.write(line1)
    f.write(line2)

f = open("validation_target_images_painting_3.txt","w")
for idx, class_ in enumerate(labeled_target_images_painting):
    line0 = str("painting/" + class_list[idx] + "/" + labeled_target_images_painting[idx][3]) + " " + str(idx) + "\n"
    line1 = str("painting/" + class_list[idx] + "/" + labeled_target_images_painting[idx][4]) + " " + str(idx) + "\n"
    line2 = str("painting/" + class_list[idx] + "/" + labeled_target_images_painting[idx][5]) + " " + str(idx) + "\n"
    f.write(line0)
    f.write(line1)
    f.write(line2)


# Creating the evalution set 
for idx, class_ in enumerate(labeled_target_images_painting):
    final_test_target.append(labeled_target_images_painting[idx][100:300])

f = open("final_test_target_painting.txt","w")
for idx, class_ in enumerate(final_test_target):
    for img in class_:
        f.write("painting/" + class_list[idx] + "/" + img + " " + str(idx) + "\n")

# Creating the unlabeled target images for one shot and three shot
for idx, class_ in enumerate(labeled_target_images_painting):
    unlabeled_target_images_painting_1.append(labeled_target_images_painting[idx][1:100])

f = open("unlabeled_target_images_painting_1.txt","w+")
for idx,class_ in enumerate(unlabeled_target_images_painting_1):
    for img in class_:
        f.write("painting/" + class_list[idx] + "/" + img + " " + str(idx) + "\n")


for idx, class_ in enumerate(labeled_target_images_painting):
    unlabeled_target_images_painting_3.append(labeled_target_images_painting[idx][3:100])

f = open("unlabeled_target_images_painting_3.txt","w+")
for idx,class_ in enumerate(unlabeled_target_images_painting_3):
    for img in class_:
        f.write("painting/" + class_list[idx] + "/" + img + " " + str(idx) + "\n")







""""
for class_examples in labeled_target_images_painting:
    for idx,example in enumerate(class_examples):
        if idx%300 == 0:
            labeled_target_images_painting_1.append(example)


for class_examples in labeled_target_images_painting:
    for idx,example in enumerate(class_examples):
        if not idx%300 == 0:
            unlabeled_target_images_painting_1.append(example)


for class_examples in labeled_target_images_painting:
    for idx,example in enumerate(class_examples):
        if idx%300 == 0 or idx%300 == 1 or idx%300 == 2:
            labeled_target_images_painting_3.append(example)

for class_examples in labeled_target_images_painting:
    for idx,example in enumerate(class_examples):
        if not (idx%300 == 0 or idx%300 == 1 or idx%300 == 2):
            unlabeled_target_images_painting_3.append(example)



for class_examples in labeled_target_images_painting:
    for idx,example in enumerate(class_examples):
        if idx%300 == 0 or idx%300 == 1 or idx%300 == 2:
            validation_target_images_painting_3.append(example) 


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


#Hey, the in validation_target_images_painting_3.txt, all the images are from the unlabeled_target_images_painting_* itself, should I follow the same setting or should I take unseen images in the validation_target_images_painting_3.txt?