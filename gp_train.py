import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import random
import torch
import torch.nn as nn
from numpy.lib.function_base import piecewise
import zarr
from skimage import data
from skimage import filters
import gunpowder as gp
from gunpowder.torch import Train
import math

zarrname='sample_data.zarr'
#set manual seed
torch.manual_seed(888)

def train(iteration, batch_size, model, loss):
    #hard set voxel_size
    voxel_size = gp.Coordinate((5, 1, 1))
    #set inpust size in voxel
    input_size = gp.Coordinate((20,100,100) * voxel_size)
    output_size = gp.Coordinate((12,12,4) * voxel_size)

    #how much to pad
    context = (input_size - output_size)/2

    #extract data
    raw = gp.ArrayKey('RAW')
    pred = gp.ArrayKey('PRED')
    gt = gp.ArrayKey('GT')
    mask = gp.ArrayKey('MASK')

    #request certain shape of the data
    request=gp.BatchRequest()
    request.add(raw, input_size)
    request.add(gt, output_size)
    
    source = gp.ZarrSource(
            zarr_name,  # the zarr container
            {raw: 'raw'},  # which dataset to associate to the array key
            {raw: gp.ArraySpec(interpolatable=True), },  # meta-information
            {gt: 'ground_truth'},
            {gt: gp.ArraySpec(interpolatable=True)} +
            #add pad here
            gp.Pad(raw, context)

        )
    # create "pipeline" consisting only of a data source
    pipeline = source

    #rotation augmentation
    pipeline += gp.ElasticAugment(
        [5,1,1],
        [0,2,2],
        [0,math.pi/2.0],
        prob_slip=0.05,
        prob_shift=0.05,
        max_misalign=25,
        spatial_dims = 2
    ) 

    #fliping augmentation
    pipeline += gp.SimpleAugment([1,2]) 
    
    #intensity augumentation
    pipeline =+ gp.IntensityAugment(
    raw,
    scale_min=0.9,
    scale_max=1.1,
    shift_min=-0.1,
    shift_max=0.1,
    z_section_wise=True)

    #noise aumentation 
    pipeline =+ gp.NoiseAugment(
        raw
    )

    #stack batch size
    pipeline += gp.Stack(batch_size)

    #training loop
    pipeline += Train(
        model,
        loss,
        optimizer = torch.optim.Adam(model.parameters()),
        inputs = {
            'input': raw
        },
        loss_inputs = {
            0: pred,
            1:gt
        },
        outputs = {
            0: pred
        }
    )

    #for loss function
    #remember to mask the padded area when calculating the loss function

    #add prediction to the request
    request[pred] = gp.Roi(output_size * voxel_size) #not sure what to put in there

    with gp.build(pipeline) :
        batch = pipeline.request_batch(request)



