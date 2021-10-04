import zarr
import gunpowder as gp
import torch

def predict(model, raw_dataset, outfile, eval=True, output_dir = '.', raw_ak_name='raw'):
    
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
    input_size = gp.Coordinate((20,100,100) * voxel_size)
    output_size = gp.Coordinate((20,100,100) * voxel_size) 
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

    request = gp.BatchRequest()
    #request[raw] = gp.Roi((0, 0), im_size)
    request[pred] = gp.Roi((0, 0), im_size)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)
        if eval:
            pass #get metrics to compute, remember masks

    #save data