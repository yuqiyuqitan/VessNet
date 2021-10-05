#prepare dataset
import numpy as np
import h5py
import os
import math
import argparse

from numpy.core.shape_base import vstack
import zarr
import gunpowder as gp


def makeZarr(input_fname, output_dir="."):
    
    fname_raw = input_fname
    path_head, path_tail = os.path.split(input_fname)
#    print(path_tail)
    output_fname =  path_tail.split('.')
    output_fname =  output_fname[0]
    fname_gt = path_head + '/../label/' + path_tail
    zarr_name = output_fname + '.zarr'
    #read in data from h5 file
    file_raw = h5py.File(fname_raw, 'r')
    raw_data = np.array(file_raw['im']) #is this z, y, x? Yes, h5 files are formated z, y, x, augusto double checked
#     raw_data = raw_data[:15,:292,:154] # to test if stack is smaller than 20 z slices
    raw_data = raw_data.astype('float32')
    file_gt = h5py.File(fname_gt, 'r')
    raw_gt = np.array(file_gt['im'])
#    print(raw_gt[0:1, 0:100, 50:51])
#     raw_gt = raw_gt[:15,:292,:154] # to test if stack is smaller than 20 z slices
    raw_gt = raw_gt.astype('uint8')
#    print(raw_gt[0:1, 0:100, 50:51])
#     print(path_head, path_tail, fname_gt)
    
#     print(raw_data.shape, raw_gt.shape)
#     print(raw_data_small.shape, raw_gt_small.shape)

    # padding if the data z < 20, add padding on z 
    if raw_data.shape[0] < 20:
        pad = np.zeros((20-raw_data.shape[0], raw_data.shape[1], raw_data.shape[2]))
        pad2 = np.zeros((20-raw_data.shape[0], raw_data.shape[1], raw_data.shape[2]), dtype='uint8')
        pad_mask = np.ones((raw_data.shape[0], raw_data.shape[1], raw_data.shape[2]), dtype='uint8')
        raw_data = np.vstack((raw_data, pad))
        raw_gt = np.vstack((raw_gt, pad))
        # create a mask to record this padding
        raw_mask = np.vstack((pad_mask, pad2))
    else:
        raw_mask = np.ones((raw_data.shape[0], raw_data.shape[1], raw_data.shape[2]), dtype='uint8')

#     print(raw_data.shape, raw_gt.shape, raw_mask.shape)
#     print(pad2.shape)
#     print(raw_mask[14:18,0:5,0:5])
#     print(raw_data.shape[0])

    # print(raw_data.shape[2])

    #splitby the x, y axis 50%, 25%, 25% into train, test, val
    y_is = raw_data.shape[1]
    x_is = raw_data.shape[2]

    # floor(y_is/2) 

    # slicing train data
    train_raw_data = raw_data[:raw_data.shape[0], :math.ceil(y_is/2), :raw_data.shape[2]]
    train_raw_gt = raw_gt[:raw_data.shape[0], :math.ceil(y_is/2), :raw_data.shape[2]]
    train_raw_mask = raw_mask[:raw_data.shape[0], :math.ceil(y_is/2), :raw_data.shape[2]]


    test_raw_data = raw_data[:raw_data.shape[0], math.ceil(y_is/2):-1, :math.ceil(x_is/2)]
    test_raw_gt = raw_gt[:raw_data.shape[0], math.ceil(y_is/2):-1, :math.ceil(x_is/2)]
    test_raw_mask = raw_mask[:raw_data.shape[0], math.ceil(y_is/2):-1, :math.ceil(x_is/2)]


    val_raw_data = raw_data[:raw_data.shape[0], math.ceil(y_is/2):-1, math.ceil(x_is/2):-1]
    val_raw_gt = raw_gt[:raw_data.shape[0], math.ceil(y_is/2):-1, math.ceil(x_is/2):-1]
    val_raw_mask = raw_mask[:raw_data.shape[0], math.ceil(y_is/2):-1, math.ceil(x_is/2):-1]

#     print(raw_data.shape, train_raw_data.shape, test_raw_data.shape, val_raw_data.shape)
#     print(output_fname)
    
#    print(train_raw_gt[0:1, 50:51, 0:100])
#    print(not os.path.exists(output_dir+'/train/'))

    if not os.path.exists(output_dir+'/train/'): os.makedirs(output_dir+'/train/')
    if not os.path.exists(output_dir+'/test/'): os.makedirs(output_dir+'/test/')
    if not os.path.exists(output_dir+'/val/'): os.makedirs(output_dir+'/val/') 


    

    f = zarr.open(output_dir+'/train/'+zarr_name, 'w')
    f['raw'] = train_raw_data.astype('float32')
    f['raw'].attrs['resolution'] = (5,1,1)
    f['gt'] = train_raw_gt.astype('uint8')
    f['gt'].attrs['resolution'] = (5,1,1)
    f['mask'] = train_raw_mask.astype('uint8')
    f['mask'].attrs['resolution'] = (5,1,1)


    f = zarr.open(output_dir+'/test/'+zarr_name, 'w')
    f['raw'] = test_raw_data.astype('float32')
    f['raw'].attrs['resolution'] = (5,1,1)
    f['gt'] = test_raw_gt.astype('uint8')
    f['gt'].attrs['resolution'] = (5,1,1)
    f['mask'] = test_raw_mask.astype('uint8')
    f['mask'].attrs['resolution'] = (5,1,1)


    f = zarr.open(output_dir+'/val/'+zarr_name, 'w')
    f['raw'] = val_raw_data.astype('float32')
    f['raw'].attrs['resolution'] = (5,1,1)
    f['gt'] = val_raw_gt.astype('uint8')
    f['gt'].attrs['resolution'] = (5,1,1)
    f['mask'] = val_raw_mask.astype('uint8')
    f['mask'].attrs['resolution'] = (5,1,1)

    #store image in zarr container
    #train/raw + gt
    #test/raw + gt
    #val/raw + gt


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "make Zarr directories for deepVess, you need train, test and val dirs")
	parser.add_argument('input_fname', type = str)
	parser.add_argument('--output_dir', type = str,default='.')
	args = parser.parse_args()
	makeZarr(args.input_fname, args.output_dir)










