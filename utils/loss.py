import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Function
import torch.nn as nn

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd
    
    @staticmethod
    def forward(self, x):
        return x.view_as(x)
    
    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) /
                    (1.0 + np.exp(-alpha * iter_num / max_iter)) -
                    (high - low) + low)


def entropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=-eta)
    out_t1 = F.softmax(out_t1,dim=1)
    loss_ent = -lamda * torch.mean(torch.sum(out_t1 *
                                             (torch.log(out_t1 + 1e-5)), 1))
    return loss_ent


def adentropy(F1, feat, lamda, weight_list = None, class_dist_threshold_list = None, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=eta)
    # Using the threshold method
    if class_dist_threshold_list is not None:
        # predicted class of the output
        pred_class_list = out_t1.data.max(1)[1]
        out_t1_max_vals = out_t1.data.max(1)[0]
        # weights for each of the samples for entropy maximization step
        weight_list = []
        for pred, val in zip(pred_class_list, out_t1_max_vals):
            if(val < class_dist_threshold_list[int(pred.cpu().data.item())]):
                weight_list.append(2)
            else:
                weight_list.append(1) 
        weight_list = torch.tensor(np.array(weight_list)).double().cuda()


        out_t1 = F.softmax(out_t1,dim=1)
        
        loss_adent = lamda * torch.mean(weight_list * torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1).double())
        return loss_adent
    # Using the image weighing method
    if weight_list is not None:
        weight_list = torch.tensor(np.array(weight_list)).double().cuda()
        out_t1 = F.softmax(out_t1,dim=1) 
        loss_adent = lamda * torch.mean(weight_list * torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1).double())
        return loss_adent
    
    # the standard loss calculation procedure without any weigths for the samples 
    out_t1 = F.softmax(out_t1,dim=1)
    loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                              (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none') # important to add reduction='none' to keep per-batch-item loss
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean() # mean over the batch
        return focal_loss

# Below given is the focal loss after incorporating the weights for each of the classes

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class CBFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(CBFocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
