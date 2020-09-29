from __future__ import print_function
import argparse
import os
import torch
from model.resnet import resnet34, resnet50
from torch.autograd import Variable
from tqdm import tqdm
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep, Predictor_attributes, Predictor_deep_attributes
from utils.return_dataset import return_dataset_test_unseen
import numpy as np
import torch.nn as nn

# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--step', type=int, default=1000, metavar='step',
                    help='loading step')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda',
                    help='dir to save checkpoint')
parser.add_argument('--method', type=str, default='MME',
                    choices=['S+T', 'ENT', 'MME'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         'S+T is training only on labeled examples')
parser.add_argument('--output', type=str, default='./output.txt',
                    help='path to store result file')
parser.add_argument('--net', type=str, default='resnet34', metavar='B',
                    help='which network ')
parser.add_argument('--source', type=str, default='Art', metavar='B',
                    help='board dir')
parser.add_argument('--target', type=str, default='Clipart', metavar='B',
                    help='board dir')
parser.add_argument('--dataset', type=str, default ='multi',
                    choices=['multi','office_home'], help='the name of dataset, multi is large scale dataset')
parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--dim', type=int, default=300,
                        help='dimensionality of the feature vector - make sure this in sync with the dim of the semantic attribute vector') 
parser.add_argument('--loss', type=str, default="CE",
                        help='loss function during model evaluation') 

device = "cuda"
args = parser.parse_args()
print('dataset %s source %s target %s network %s' %
      (args.dataset, args.source, args.target, args.net))
target_loader_unl, class_list = return_dataset_test_unseen(args)
use_gpu = torch.cuda.is_available()

if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == 'resnet50':
    G = resnet50()
    inc = 2048
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')


if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list),inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc, temp=args.T)

"""
if "resnet" in args.net:
    F1 = Predictor_deep_attributes(num_class=len(class_list),inc=inc)
else:
    F1 = Predictor_attributes(num_class=len(class_list), inc=inc, temp=args.T)
"""

# Loading the model weights from the checkpoint
filename = "save_model_ssda/ours.ckpt.best.pth.tar"
main_dict = torch.load(filename)
args.step = main_dict['step']
print("Inferencing is being done with model at step: ", args.step)
print("best accuracy, ", main_dict['best_acc_test'])
print(filename)
G.cuda()
F1.cuda()
G.load_state_dict(main_dict['G_state_dict'])
F1.load_state_dict(main_dict['F1_state_dict'])

im_data_t = torch.FloatTensor(1)
gt_labels_t = torch.LongTensor(1)

im_data_t = im_data_t.cuda()
gt_labels_t = gt_labels_t.cuda()

im_data_t = Variable(im_data_t)
gt_labels_t = Variable(gt_labels_t)

if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)


"""
def eval(loader, output_file="output.txt"):
    G.eval()
    F1.eval()
    size = 0
    with open(output_file, "w") as f:
        with torch.no_grad():
            for batch_idx, data_t in tqdm(enumerate(loader)):
                im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
                gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
                paths = data_t[2]
                feat = G(im_data_t)
                output1 = F1(feat)
                size += im_data_t.size(0)
                pred1 = output1.data.max(1)[1]
                for i, path in enumerate(paths):
                    f.write("%s %d\n" % (path, pred1[i]))


eval(target_loader_unl, output_file="%s_%s_%s.txt" % (args.method, args.net,
                                                      args.step))


"""

def test(loader):
        G.eval()
        F1.eval()
        test_loss = 0
        correct = 0
        size = 0
        num_class = len(class_list)
        output_all = np.zeros((0, num_class))

        # Setting the loss function to be used for the classification loss
        if args.loss == 'CE':
            criterion = nn.CrossEntropyLoss().to(device)
        if args.loss == 'FL':
            criterion = FocalLoss(alpha = 1, gamma = args.gamma).to(device)
        if args.loss == 'CBFL':
            # Calculating the list having the number of examples per class which is going to be used in the CB focal loss
            beta = args.beta
            effective_num = 1.0 - np.power(beta, class_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(class_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
            criterion = CBFocalLoss(weight=per_cls_weights, gamma=args.gamma).to(device)

        confusion_matrix = torch.zeros(num_class, num_class)
        with torch.no_grad():
            for batch_idx, data_t in enumerate(loader):
                im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
                gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
                feat = G(im_data_t)
                output1 = F1(feat)
                output_all = np.r_[output_all, output1.data.cpu().numpy()]
                size += im_data_t.size(0)
                pred1 = output1.data.max(1)[1]
                for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                correct += pred1.eq(gt_labels_t.data).cpu().sum()
                test_loss += criterion(output1, gt_labels_t) / len(loader)
        np.save("cf_target.npy",confusion_matrix)
        #print(confusion_matrix)
        print('\nTest set: Average loss: {:.4f}, '
            'Accuracy: {}/{} F1 ({:.4f}%)\n'.
            format(test_loss, correct, size,
                    100. * correct / size))
        return test_loss.data, 100. * float(correct) / size

_,acc = test(target_loader_unl)
