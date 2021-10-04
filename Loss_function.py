#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import Dataset, DataLoader

import itk
import sys
from skimage import metrics
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


assert torch.cuda.is_available()
device = torch.device("cuda")


# In[ ]:


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


# In[ ]:


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
        return (2 * intersection / denominator.clamp(min=self.eps))


# In[ ]:


BCEloss = WeightedBCELoss()
Diceloss = WeightedDiceCoefficient()
loss = BCEloss(prediction, groundtruth, weight_mask) + Diceloss(prediction, groundtruth, weight_mask)

