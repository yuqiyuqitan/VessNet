from math import e
from torch.utils.tensorboard import SummaryWriter
import torch
from VessNet_architecture import VessNet
import gunpowder as gp
from gp_train import get_pipeline
from gp_predict import get_pred_pipeline
from gp_predict_noloop import get_pred_pipeline_nl
from imshow import imshow
from loss import loss_combined, loss_bce, loss_dice
import logging
import os
import zarr
import numpy as np

logging.basicConfig(level = logging.INFO)
gp.PrintProfilingStats(every=500)

#assert torch.cuda.is_available() #not sure
#device = torch.device("cuda")

train_dir = "/mnt/efs/woods_hole/danceParty/tiff_images/03_zarr_train/train/"
val_dir = "/mnt/efs/woods_hole/danceParty/tiff_images/03_zarr_train/val/"
checkpoint = "model_checkpoint_100"
train = False
view = False
prediction = True
# set manual seed
# torch.manual_seed(888)

# hard set voxel_size
voxel_size = gp.Coordinate((5, 1, 1))
# set inpust size in voxel
input_size = gp.Coordinate((20, 128, 128))
output_size = gp.Coordinate((20, 128, 128))
model = VessNet()
loss = loss_combined

iterations = 1000000

raw = gp.ArrayKey("RAW")
gt = gp.ArrayKey("GT")
mask = gp.ArrayKey("MASK")
pred = gp.ArrayKey("PRED")

#This view 
if view:
    #this generate snapshoot for rendering
    pipeline = get_pipeline(raw_data = raw_dataset, input_size = input_size, output_size = output_size, train=False)
    request = gp.BatchRequest()
    request.add(raw, input_size * voxel_size)
    request.add(gt, output_size * voxel_size)
    request.add(mask, output_size *  voxel_size)
    with gp.build(pipeline):
        batch = pipeline.request_batch(request)
#this train
if train:
    pipeline = get_pipeline(train_dir = train_dir, model = model, loss=loss, input_size = input_size, output_size = output_size, train=True)
    request = gp.BatchRequest()
    request.add(raw, input_size * voxel_size)
    request.add(gt, output_size * voxel_size)
    request.add(mask, output_size *  voxel_size)
    request.add(pred, output_size *  voxel_size)
    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

if prediction:
    fnames = os.listdir(val_dir)
    for fname in fnames:
        tmp = zarr.open(os.path.join(val_dir, fname),'r')
        context = (output_size - input_size) / 2
        source = gp.ZarrSource(
            os.path.join(val_dir, fname),  # the zarr container
            {
                raw: "raw",
            },  # which dataset to associate to the array key
            {
                raw: gp.ArraySpec(interpolatable=True),  # meta-information
            })
        with gp.build(source):
            total_input_roi = source.spec[raw].roi.grow(context, context)
            total_output_roi = source.spec[raw].roi
        pipeline = get_pred_pipeline_nl(val_dir = val_dir, val_fname = fname, model = model, checkpoint=checkpoint, input_size = input_size, output_size = output_size, output_dir="")
        request = gp.BatchRequest()
        request.add(raw, total_input_roi.get_shape())
        request.add(pred, total_output_roi.get_shape())
        with gp.build(pipeline):
            batch = pipeline.request_batch(request)




    