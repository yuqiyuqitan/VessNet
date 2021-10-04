import zarr
import gunpowder as gp
import torch

def predict(model, raw_dataset, outfile, eval=True):
    
    #if i am using this for evaluation
    if eval:
        model.eval()
        mask = gp.ArrayKey('MASK')
        gt = gp.ArrayKet('GT')

    #create keys to keep track of the data
    raw = gp.ArrayKey('RAW')
    pred = gp.ArrayKey('PRED')

    voxel_size = gp.Coordinate((5, 1, 1))
    input_size = gp.Coordinate((20,100,100) * voxel_size)
    output_size = gp.Coordinate((4,12,12) * voxel_size)
    #how much to pad
    context = (input_size - output_size)/2

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(pred, output_size)
    scan_request.add(gt, output_size)

    source = tuple(gp.ZarrSource(
            zarr_name,  # the zarr container
            {raw: 'raw'},  # which dataset to associate to the array key
            {raw: gp.ArraySpec(interpolatable=True)}) +  # meta-information)
            gp.Pad(raw, context) +
            gp.RandomLocation()
    )

    pipeline = source
