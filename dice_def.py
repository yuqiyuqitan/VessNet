from torch.utils.data import Dataset, DataLoader

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


# Give it validation data as DataLoader class

def getDiceCoefficient(model, loader):
    # reinitializing metric values
    val_metric = 0
    # disable gradients during validation
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            val_metric += metric(prediction, y).item()
    
    val_metric /= len(loader)
    return val_metric

'''
Usage:
net <- is defined already
data_is <- is defined already

loader_is = DataLoader(data_is, batch_size=5)
getDiceCoefficient(net, loader_is)
'''
