import zarr
import gunpowder as gp
import torch

def predict(model, raw_dataset, outfile):
    
    
    #create keys to keep track of the data
    raw = gp.ArrayKey('RAW')
    pred = gp.ArrayKey('PRED')
    mask = gp.ArrayKey('MASK')

    voxel_size = gp.Coordinate((5, 1, 1))
    input_size = gp.Coordinate((20,100,100) * voxel_size)
    output_size = gp.Coordinate((12,12,4) * voxel_size)

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(pred, output_size)

    source = gp.ZarrSource(
            zarr_name,  # the zarr container
            {raw: 'raw'},  # which dataset to associate to the array key
            {raw: gp.ArraySpec(interpolatable=True), },  # meta-information
            gp.Pad(raw, context)

    )

    pipeline = source
