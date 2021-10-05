import zarr
import gunpowder as gp
import torch
import os
import glob
from loss import loss_combined, loss_bce, loss_dice
import logging
from VessNet_architecture import VessNet

logging.basicConfig(level = logging.INFO)
gp.PrintProfilingStats(every=500)

def get_pred_pipeline(
    model,
    val_dir,
    input_size,
    output_size,
    voxel_size = gp.Coordinate((5, 1, 1)),
    eval=True,
    checkpoint_dir = None,
    output_dir=".",
    #raw_ak_name="raw"
):
    # set the model in evaluation mode
    model.eval()

    # create keys to keep track of the data
    raw = gp.ArrayKey("RAW")
    pred = gp.ArrayKey("PRED")

    # if i am using this for validation
    if eval:
        gt = gp.ArrayKey("GT")
        mask = gp.ArrayKey("MASK")

    
    input_size = input_size * voxel_size
    output_size = output_size * voxel_size
    # how much to pad
    context = (input_size - output_size) / 2

    fnames = os.listdir(val_dir)

    for fname in fnames:
        tmp = zarr.open(os.path.join(val_dir, fname), "r")
        im_size = tmp["mask"].shape
        # create variable to call
        scan_request = gp.BatchRequest()
        scan_request.add(raw, input_size)
        scan_request.add(pred, output_size)
        
        if eval:
            scan_request.add(gt, output_size)
            scan_request.add(mask, output_size)        
            source = gp.ZarrSource(
                        os.path.join(val_dir,fname),  # the zarr container
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
            source += gp.Pad(raw, None) + gp.Pad(mask, context) + gp.Pad(gt, context)
        else:
            source = gp.ZarrSource(
                        os.path.join(val_dir,fname),  # the zarr container
                        {raw: "raw",}, 
                        {raw: gp.ArraySpec(interpolatable=True)})
            source += gp.Pad(raw, None) + gp.Pad(mask, context) + gp.Pad(gt, context)

        pipeline = source
        
        if eval:
            pipeline += gp.Unsqueeze([raw, gt, mask],0)
            pipeline += gp.Unsqueeze([raw, gt, mask],0)
        else:
            pipeline += gp.Unsqueeze([raw],0)
            pipeline += gp.Unsqueeze([raw],0)
        
        # predict
        # add a predict node
        if eval:
            pipeline += gp.torch.Predict(
                model,
                checkpoint = checkpoint_dir,
                inputs={"input": raw},
                outputs={0: pred}, #no need to output the loss?
                #loss_inputs={0: pred, 1: gt, 2: mask}
            )
            # save data
            pipeline += gp.ZarrWrite(
                {pred: "volumes/pred"},
                output_filename=os.path.join(output_dir, "pred", fname, ".zarr"),
                compression_type="gzip",
            )
        else:
            pipeline += gp.torch.Predict(
                model,
                checkpoint = checkpoint_dir,
                inputs={"input": raw},
                outputs={0: pred}, #no need to output the loss?
            )
            # save data
            pipeline += gp.ZarrWrite(
                {pred: "volumes/pred"},
                output_filename=os.path.join(output_dir, "pred", fname, ".zarr"),
                compression_type="gzip",
            )

        pipeline += gp.Scan(reference=scan_request)

        return pipeline
