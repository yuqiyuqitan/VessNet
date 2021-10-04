from torch.utils.tensorboard import SummaryWriter
import torch
import gunpowder as gp
from gp_train import get_pipeline
from imshow import imshow

raw_dataset = "sample_data.zarr"
# set manual seed
# torch.manual_seed(888)

# hard set voxel_size
voxel_size = gp.Coordinate((5, 1, 1))
# set inpust size in voxel
input_size = gp.Coordinate((20, 128, 128))
output_size = gp.Coordinate((20, 128, 128))

raw = gp.ArrayKey("RAW")
gt = gp.ArrayKey("GT")

pipeline = get_pipeline(raw_data = raw_dataset, input_size = input_size, output_size = output_size, train=False)

request = gp.BatchRequest()
request.add(raw, input_size * voxel_size)
request.add(gt, output_size * voxel_size)

with gp.build(pipeline):
    batch = pipeline.request_batch(request)