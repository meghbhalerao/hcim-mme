#from tsnecuda import TSNE
import torch
import numpy as np
import sys
sys.path.append('/cbica/home/bhaleram/comp_space/random/personal/iisc_project/MME/')
from loaders.data_list import Imagelists_VISDA, return_classlist, return_number_of_label_per_class
from model.basenet import *
from model.resnet import *
from torchvision import transforms
from utils.return_dataset import ResizeImage
from MulticoreTSNE import MulticoreTSNE as TSNE
import time
import os
#from sklearn.decomposition import PCA

# Defining return dataset function here
net = "alexnet"
root = '../data/multi/'
source = "painting"
target = "real"
n_class = 126
k = 10
image_set_file_test = "/cbica/home/bhaleram/comp_space/random/personal/iisc_project/MME/data/txt/multi/unlabeled_target_images_%s_3.txt"%(target)
model_path = "../freezed_models/alexnet_p2r_ours.ckpt.best.pth.tar"
ours = True


def get_dataset(net,root,image_set_file_test):
    if net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root,
                                            transform=data_transforms['test'],
                                            test=True)
    class_list = return_classlist(image_set_file_test)
    num_images = len(target_dataset_unl)
    if net == 'alexnet':
        bs = 1
    else:
        bs = 1

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl, batch_size=bs, num_workers=3,shuffle=False, drop_last=False)
    return target_loader_unl,class_list


target_loader_unl,class_list = get_dataset(net,root,image_set_file_test)

# Getting which classes have most and least number of samples in both the source and target combined
class_list.sort()
class_num_list = []
for class_ in class_list:
    class_num_list.append(len(os.listdir(os.path.join(root,source,class_))) + len(os.listdir(os.path.join(root,target,class_))))

sorted_list = [x for _,x in sorted(zip(class_num_list,class_list))]
print(sorted_list)
minority_k = sorted_list[0:k]
majority_k = sorted_list[n_class-k:n_class]

f = open(image_set_file_test,"r")
f_majority = open("unlabeled_target_images_%s_3_majority.txt"%(target), "w")
for line in f:
    for class_ in majority_k:
        if str(class_) in str(line):
            f_majority.write(str(line))

f_majority.close()

f = open(image_set_file_test,"r")
f_minority = open("unlabeled_target_images_%s_3_minority.txt"%(target), "w")
for line in f:
    for class_ in minority_k:
        if str(class_) in str(line):
            f_minority.write(str(line))
f_minority.close()

f.close()

target_loader_unl_minority,class_list = get_dataset(net,root,"./unlabeled_target_images_%s_3_minority.txt"%(target))
target_loader_unl_majority,class_list = get_dataset(net,root,"./unlabeled_target_images_%s_3_majority.txt"%(target))


# Deinfining the pytorch networks
if net == 'resnet34':
    G = resnet34()
    inc = 512
elif net == 'resnet50':
    G = resnet50()
    inc = 2048
elif net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')


if ours: 
    if net == 'resnet34':
        F1 = Predictor_deep_2(num_class=n_class,inc=inc,feat_dim = 50)
        print("Using: Predictor_deep_attributes")
    else:
        F1 = Predictor_attributes(num_class=n_class,inc=inc,feat_dim = 50)
        print("Using: Predictor_attributes")

else:
    if net == 'resnet34':
        F1 = Predictor_deep(num_class=n_class,inc=inc)
        print("Using: Predictor_deep")
    else:
        F1 = Predictor(num_class=n_class, inc=inc, temp=0.05)
        print("Using: Predictor")


G.cuda()
F1.cuda()

checkpoint = torch.load(model_path)
print(checkpoint['step'])
#f = open("/cbica/home/bhaleram/comp_space/random/personal/iisc_project/MME/entropy_weight_files/MME_alexnet_real_painting_0.txt")


# Loading the weights from the checkpoint
G.load_state_dict(checkpoint["G_state_dict"])
#F1.load_state_dict(checkpoint["F1_state_dict"])

features = []
labels = []
weights = []

# Read the weight file and converting it to a dictionary
#weight_dict = {}
#for line in f:
#    line_list = line.split()
#    weight_dict[line_list[1]] = float(line_list[0])


start = time.time()
with torch.no_grad():
    for idx, image_list in enumerate(target_loader_unl_minority):
        #print(image_list)
        image = image_list[0].cuda()
        label = image_list[1].cpu().data.item()
        img_path = image_list[2][0]
        output = G(image)
        #output = F1.fc1(output)
        #print(output.shape)
        output = output.cpu().numpy()
        output = output[0]
        features.append(output)
        labels.append(label)
        print(idx)
        #weights.append(weight_dict[img_path])

end = time.time()
print((end-start)/60)



#prototype = F1.fc2.weight.cpu().detach().numpy()

#for i in range(45):
#    features.append(prototype[i,:])

#for i in range(45):
#    labels.append(-100)


features = np.array(features)
print(features.shape)
np.save('features.npy', features)
print(len(labels))
labels = np.array(labels)
np.save('labels.npy',labels)
#print(len(weights))
#weights = np.array(weights)
#np.save('weights.npy', weights)


#features = np.load("features.npy")
#labels = np.load("labels.npy")
#print(features.shape,labels.shape)
#pca = PCA(n_components=50)
#features = pca.fit_transform(features) 


tsne = TSNE(perplexity = 30, n_jobs=1, n_iter = 3000, verbose=1)
X_embedded = tsne.fit_transform(features)
np.save('tsne_embeddings.npy', X_embedded)
print((end-start)/60)