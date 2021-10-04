
#how to log things outside of gunpowder
from torch.utils.tensorboard import SummaryWriter
#import torch
#from torch.utils.tensorboard import tensorboar

from gp_train import pipeline, input_size, raw
from imshow import imshow
"""
#log output

        #log loss to tensor board
step = epoch * iteration * batch_id #define step
tb_logg.add_add_scalar(
    tag = "train_loss", scalar_value = loss.item(), global_step = step #need to change this, cuz we have more loss functiosn
)

#log_image_step = 20
#load image to tensor 
if step % log_image_step == 0:
        tb_log.add_images(tag = 'input', img_tensor = x.to("cpu"), global_step = step)
        tb_log.add_images(tag = 'target', img_tensor = y.to("cpu"), global_step = step)     
        tb_log.add_images(tag = 'prediction', img_tensor = pred.to("cpu").detach(), global_step = step)  #check variable
"""

#request a snap shot to see if the augmentation makes sense
snapshot_request = gp.BatchRequest()
snapshot_request.add(raw, input_size)

with gp.build(pipeline):
    batch = pipeline.request_batch(snapshot_request)

    imshow(batch[raw].data)