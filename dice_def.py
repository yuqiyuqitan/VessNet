from torch.utils.data import Dataset, DataLoader

import itk
import sys
from skimage import metrics
import matplotlib.pyplot as plt
import numpy as np

assert torch.cuda.is_available()
device = torch.device("cuda")

# sorensen dice coefficient implemented in torch
# the coefficient takes values in [0, 1], where 0 is
# the worst score, 1 is the best score
class DiceCoefficient(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        
    # the dice coefficient of two sets represented as vectors a, b ca be 
    # computed as (2 *|a b| / (a^2 + b^2))
    def forward(self, prediction, target):
        intersection = (prediction * target).sum()
        denominator = (prediction * prediction).sum() + (target * target).sum()
        return (2 * intersection / denominator.clamp(min=self.eps))

    
# EvaluateSegmentation truth.nii segment.nii –use all  -thd 0.5  

# Give it data as DataLoader class for now, but this will change because gunpowder

dice_metric = DiceCoefficient()
# metrics.hausdorff_distance(coords_a, coords_b)

# Average hausdorff distance
# Balanced average hausdorff distance

# Returns a dictionary with average metrics with the keys: 'dice'
def Getmetrics(model, loader, dice_metric):
    # reinitializing metric values
    dice_is = 0
    hausdorff_distance_is = 0
    # disable gradients during validation
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
#             print(x.shape, y.shape, prediction.shape)
            dice_is += dice_metric(prediction, y).item() #adds all dice together
    
#             Hard coded 0.5 threshold
            prediction_binary = prediction.cpu() > 0.5
            y_binary = y.cpu() > 0.5
#             print(prediction_binary.shape)
#             print(prediction_binary.shape, y.cpu().shape)
#             print(y[(0,0,0,0)])

            if len(x) == 0:
                print("Error x len: ", len(x), ", y len: ", len(y))
                continue
            elif len(y) == 0:
                print("Error x len: ", len(x), ", y len: ", len(y))
                continue
            
            prediction_binary = prediction_binary.numpy()
            y_binary = y_binary.numpy()
            prediction_binary_nonzero_len = len(np.transpose(np.nonzero(prediction_binary)))
            b_points_nonzero_len = len(np.transpose(np.nonzero(y_binary)))
            
            if prediction_binary_nonzero_len == 0 or b_points_nonzero_len == 0:
                print("Inf avoided for hausdorff_distance")
                continue
    
            hausdorff_distance_is += metrics.hausdorff_distance(prediction_binary, y_binary) # adds all hausdorff together
            #             print(hausdorff_distance_is)
    
#     print("Distance is: ",hausdorff_distance_is, len(loader))
    dice_is /= len(loader) # gets mean dice
    hausdorff_distance_is /= len(loader) #computes mean hausdorff
    
    metrics_are = {'dice': dice_is, 'hausdorff' : hausdorff_distance_is}
    return metrics_are
"""
Usage:
net <- is defined already
data_is <- is defined already, not sure yet what method will be used to make a data loader

loader_is = DataLoader(data_is, batch_size=5), won't be used because gunpowder
Getmetrics(net, loader_is, dice_metric)
"""

