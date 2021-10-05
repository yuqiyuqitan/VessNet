import torch
import sys
from skimage import metrics
import numpy as np
import torch.nn as nn

# calculate weighted binary cross-entropy loss
class WeightedBCELoss(torch.nn.BCELoss):

    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, prediction, groundtruth, weight_mask):
        prediction = prediction * weight_mask
        groundtruth = groundtruth * weight_mask
        weighted_BCEloss = super(WeightedBCELoss, self).forward(
                prediction,
                groundtruth)        
        return weighted_BCEloss

# calculate weighted dice loss
class WeightedDiceCoefficient(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps        
    # computed as (2 *|a b| / (a^2 + b^2))
    def forward(self, prediction, groundtruth, weight_mask):
        prediction = prediction * weight_mask
        groundtruth = groundtruth * weight_mask
        intersection = (prediction * groundtruth).sum()
        denominator = (prediction * prediction).sum() + (groundtruth * groundtruth).sum()
        return 1-(2 * intersection / denominator.clamp(min=self.eps))

def loss(prediction, groundtruth, weight_mask):
    BCEloss = WeightedBCELoss()
    Diceloss = WeightedDiceCoefficient()
    groundtruth = groundtruth.float()
    weight_mask = weight_mask.float()
    #print(prediction.type(), groundtruth.type(), weight_mask.type())
    loss = BCEloss (prediction = prediction, groundtruth = groundtruth, weight_mask=weight_mask)+ \
        Diceloss (prediction = prediction, groundtruth = groundtruth, weight_mask=weight_mask)
    return loss