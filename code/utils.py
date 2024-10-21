# utils.py

# This file contains utility functions for the project.
# This file should include:
# - Loss computation (e.g., custom loss functions)
# - Metrics computation (e.g., IoU, mIoU)
# - Logging and visualization tools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1) # N,C,H,W => N,C,H*W
            input = input.transpose(1,2) # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2)) # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        target = target.long()
        logpt = F.log_softmax(input, dim = 1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        return loss.mean()

class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss2d, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, target):
        target = target.long()
        return self.criterion(inputs, target.squeeze(1))

# consistency constraint loss
class ConsistencyConstraintLoss(nn.Module):
    def __init__(self):
        super(ConsistencyConstraintLoss, self).__init__()

    def forward(self, original_output, augmented_output, T = 1):
        # use KL divergence to calculate the consistency constraint loss
        KD_loss = nn.KLDivLoss()(F.log_softmax(augmented_output/T, dim=1), 
                             F.softmax(original_output/T, dim=1)) * T * T
        return KD_loss

# comprehensive loss: cross entropy + boundary loss + consistency constraint loss
class TotalLoss(nn.Module):
    def __init__(self, boundary_weight=0.1, consistency_weight=2):
        super(TotalLoss, self).__init__()
        self.consistency_loss = ConsistencyConstraintLoss()
        self.truth_loss = CrossEntropyLoss2d()
        self.focal_loss = FocalLoss(gamma=2)
        self.boundary_weight = boundary_weight
        self.consistency_weight = consistency_weight
    def forward(self, prediction_mask, prediction_edge, target_mask, target_edge, original_output_mask, augmented_output_mask):
        loss_truth = self.truth_loss(prediction_mask, target_mask)
        loss_edge = self.focal_loss(prediction_edge, target_edge)
        # calculate the consistency constraint loss
        loss_consistency1 = self.consistency_loss(Variable(original_output_mask, requires_grad=False), augmented_output_mask)
        # comprehensive loss
        total_loss = loss_truth + self.boundary_weight * loss_edge + self.consistency_weight * loss_consistency1
        return total_loss

# evaluation metric: IoU (Intersection over Union)
def compute_iou(prediction, target):
    # convert prediction to binary mask
    # prediction = torch.argmax(prediction, dim=1).unsqueeze(1)  # now prediction shape is B * 1 * h * w
    prediction = F.softmax(prediction, dim=1)  # apply softmax
    prediction = (prediction[:, 1:] > 0.5).float()  # greater than 0.5 is 1, less than 0.5 is 0
    # ensure the input is numpy array
    prediction = prediction.cpu().numpy()
    target = target.cpu().numpy()

    sum1 = prediction + target
    sum1[sum1 > 0] = 1
    sum2 = prediction + target
    sum2[sum2 < 2] = 0
    sum2[sum2 >= 2] = 1
    
    # calculate the IoU of each sample
    iou_scores = []
    for i in range(prediction.shape[0]):
        # print(np.sum(sum1[i]))
        # print(np.sum(sum2[i]))
        if np.sum(sum1[i]) == 0:
            iou_scores.append(1.0)
        else:
            iou_scores.append(np.sum(sum2[i]) / np.sum(sum1[i]))

    # return the mean IoU
    return np.mean(iou_scores)

def multi_class_loss(prediction, target):
    criterion = nn.CrossEntropyLoss()
    target = target.squeeze(1).long()
    return criterion(prediction, target)

def compute_iou_face(prediction, target, num_classes=9):
    # prediction: B * 9 * h * w
    # target: B * 1 * h * w
    device = prediction.device
    prediction = torch.argmax(prediction, dim = 1).squeeze(1)
    target = target.squeeze(1)
    # initialize intersection and union tensor
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    
    # calculate the intersection and union of each class
    for i in range(num_classes):
        pred_mask = (prediction == i)
        target_mask = (target == i)
        intersection[i] = (pred_mask & target_mask).float().sum()
        union[i] = (pred_mask | target_mask).float().sum()
    
    # calculate IoU
    iou = intersection / (union + 1e-6)
    
    # return the mean IoU
    return iou.mean().item()

class F_total_loss(nn.Module):
    def __init__(self, consistency_weight=2):
        super(F_total_loss, self).__init__()
        self.consistency_loss = ConsistencyConstraintLoss()
        self.consistency_weight = consistency_weight
        
    def forward(self, prediction_mask, target_mask, original_output_mask, augmented_output_mask):
        loss_truth = multi_class_loss(prediction_mask, target_mask)
        # calculate the consistency constraint loss
        loss_consistency = self.consistency_loss(Variable(original_output_mask, requires_grad=False), augmented_output_mask)
        # comprehensive loss
        total_loss = loss_truth + self.consistency_weight * loss_consistency
        return total_loss
    