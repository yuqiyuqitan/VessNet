import zarr
import gunpowder as gp
import torch
import os

def predict(model, raw_dataset, out_fname = 'predict.hdf5', eval=True, output_dir = '.', raw_ak_name='raw'):
    #set the model in evaluation mode
    model.eval()

    #create keys to keep track of the data
    raw = gp.ArrayKey('RAW')
    pred = gp.ArrayKey('PRED')
    
    #if i am using this for validation
    if eval:
        gt = gp.ArrayKet('GT')    
        mask = gp.ArrayKey('MASK')

    im_size = zarr.open(raw_dataset,'r')[raw_ak_name].shape
    voxel_size = gp.Coordinate((5, 1, 1))
    input_size = gp.Coordinate((20,128,128)) * voxel_size
    output_size = gp.Coordinate((20,128,128)) * voxel_size 
    #how much to pad
    context = (input_size - output_size)/2

    #create variable to call
    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(pred, output_size)

    if eval:
        scan_request.add(gt, output_size)

    #load data and add padding
    if eval:
        source=gp.ZarrSource(
                raw_dataset,  # the zarr container
                {raw: 'raw'},  # which dataset to associate to the array key
                {raw: gp.ArraySpec(interpolatable=True)},
                {gt: 'gt'},
                {gt: gp.ArraySpec(interpolatable=False)},
                {mask: 'mask'},
                {mask: gp.ArraySpec(interpolatable=False)})
        source += gp.Pad(raw, None) + gp.Pad(mask, context) + gp.Pad(gt, context)            
    else:
        source=gp.ZarrSource(
                raw_dataset,  # the zarr container
                {raw: 'raw'},  # which dataset to associate to the array key
                {raw: gp.ArraySpec(interpolatable=True)})
        source += gp.Pad(raw, context)         

    #random_location = gp.RandomLocation()

    pipeline = source

    #predict 
    # add a predict node
    pipeline += gp.tensorflow.Predict(
        model,
        input = {
            'input':raw
        },
        outputs = {
            0:pred
        }
    )


    #save data
    pipeline += gp.Hdf5Write(
        {
            pred: 'volumes/pred'
        },
        output_filename = os.join.Path(output_dir, out_fname),
        compression_type = 'gzip'
    )

    #save how much time was spent
    request = gp.BatchRequest()
    #request[raw] = gp.Roi((0, 0), im_size)
    request[pred] = gp.Roi((0, 0), im_size)

    #iterate through the whole dataset
    pipeline += gp.Scan(reference = request)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)
        if eval:
            pass #get metrics to compute, remember masks
