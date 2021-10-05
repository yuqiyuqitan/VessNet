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
from noise_augment_node import NoiseAugmentRange
from train_node import TrainImageTb
import glob
import time

def get_pipeline(train_dir, input_size, output_size, model = None, loss=None, voxel_size = gp.Coordinate((5, 1, 1)), train = False, save_every=5, iteration = 10, batch_size=5):
    # set the model to be in the training mode
    if train:
        model.train()

    input_size = input_size * voxel_size
    output_size = output_size * voxel_size
    # extract data
    raw = gp.ArrayKey("RAW")
    pred = gp.ArrayKey("PRED")
    gt = gp.ArrayKey("GT")
    mask = gp.ArrayKey("MASK")
    context = (input_size - output_size) / 2

    # request certain shape of the data
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(pred, output_size)
    request.add(gt, output_size)
    request.add(mask, output_size)

    fnames = os.listdir(train_dir)
    sizes = []
    
    fnames_deepVess = glob.glob(os.path.join(train_dir, "001*.zarr"))   
    for fname in fnames:
        tmp = zarr.open(os.path.join(train_dir, fname), 'r')
        size = np.sum(tmp['mask'])
        if os.path.join(train_dir, fname) in fnames_deepVess:
            size /= len(fnames_deepVess)
        sizes.append(size)
    
    print("Load data")
    source = tuple(gp.ZarrSource(
        os.path.join(train_dir,raw_data),  # the zarr container
        {
            raw: "raw",
            gt: "gt",
            mask: "mask",
        },  # which dataset to associate to the array key
        {
            raw: gp.ArraySpec(interpolatable=True),  # meta-information
            gt: gp.ArraySpec(interpolatable=False),
            mask: gp.ArraySpec(interpolatable=False)
        })
        + gp.Pad(raw, None)
        + gp.Pad(gt, context)
        + gp.Pad(mask, context)
        + gp.RandomLocation()  # create random location 
        for raw_data in fnames
    )
    # randomly choose a sample
    pipeline = source

    # choose sample based on probablities
    pipeline += gp.RandomProvider(probabilities=sizes)

    print("Start augmentation")
    # rotation augmentation
    elastic_augment = gp.ElasticAugment(
        [2, 10, 10],
        0,  # potentially add elastic deformation?
        [0, math.pi / 2.0],
        prob_slip=0,
        prob_shift=0,
        max_misalign=0,
        spatial_dims=2,
    )

    # fliping augmentation
    simple_augmentation = gp.SimpleAugment([1,2], [1,2])

    # intensity augumentation
    intensity_augmentation = gp.IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.1,
        shift_max=0.1,
        z_section_wise=False,
    )

    # noise aumentation
    noise_augment = NoiseAugmentRange(raw)

    # scale augumentation (resolution) ? 

    # complete the augmentation
    pipeline = pipeline + simple_augmentation + intensity_augmentation + noise_augment

    # stack batch size
    pipeline += gp.Stack(batch_size)

    pipeline += gp.Unsqueeze([raw, gt, mask],1)

    if train:
        print("Start training")
        # training loop
        pipeline += TrainImageTb(
            model,
            loss,
            show_log_im = True,
            optimizer=torch.optim.Adam(model.parameters()),
            inputs={
                "input": raw,
            },
            loss_inputs={0: pred, 1: gt, 2: mask},
            outputs={0: pred},
            save_every = 400,
            log_every = 10
        )
        pipeline += gp.Snapshot(
            dataset_names = {raw: "raw", gt: "gt", pred: "pred"},
            output_dir='train_snapshot',
            output_filename = 'batch_{iteration}.zarr',
            every = 100
        )
    else:    
        pipeline += gp.Snapshot(
            dataset_names = {raw: "raw_aug", gt: "gt_aug"},
            output_filename = 'batch_{id}.zarr'
        )

    return pipeline

