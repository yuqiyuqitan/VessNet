from torch.utils.tensorboard import SummaryWriter
import torch
from VessNet_architecture import VessNet
import gunpowder as gp
from gp_train import get_pipeline
from imshow import imshow
from loss import loss

#assert torch.cuda.is_available() #not sure
#device = torch.device("cuda")

train_dir = "/mnt/efs/woods_hole/danceParty/tiff_images/03_zarr_train/train/"
# set manual seed
# torch.manual_seed(888)

# hard set voxel_size
voxel_size = gp.Coordinate((5, 1, 1))
# set inpust size in voxel
input_size = gp.Coordinate((20, 128, 128))
output_size = gp.Coordinate((20, 128, 128))

raw = gp.ArrayKey("RAW")
gt = gp.ArrayKey("GT")
mask = gp.ArrayKey("MASK")
pred = gp.ArrayKey("PRED")

#this generate snapshoot for rendering
#pipeline = get_pipeline(raw_data = raw_dataset, input_size = input_size, output_size = output_size, train=False)

model = VessNet()

#this train
pipeline = get_pipeline(train_dir = train_dir, model = model, loss=loss, input_size = input_size, output_size = output_size, train=True)

request = gp.BatchRequest()
request.add(raw, input_size * voxel_size)
request.add(gt, output_size * voxel_size)
request.add(mask, output_size *  voxel_size)
request.add(pred, output_size *  voxel_size)

with gp.build(pipeline):
    batch = pipeline.request_batch(request)
    