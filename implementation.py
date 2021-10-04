from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.tensorboard import tensorboard
from VessNet import .

#load data

#put model here

tb_logger = SummaryWriter() #set a folder in git repo

step = 0
while step < num_epochs:
    train(net, loader, optimizer, loss_function, tb_logger, activation) #populate
    
    validate(net, loader, optimizer, loss_function, tb_logger, activation)