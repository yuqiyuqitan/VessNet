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
#    fname_gt = path_head + '/../label/' + path_tail
    zarr_name = output_fname + '.zarr'
    #read in data from h5 file
    file_raw = h5py.File(fname_raw, 'r')
    raw_data = np.array(file_raw['im']) #is this z, y, x? Yes, h5 files are formated z, y, x, augusto double checked
    raw_data = raw_data.astype('float32')
#     raw_data = raw_data[:15,:292,:154] # to test if stack is smaller than 20 z slices
#    file_gt = h5py.File(fname_gt, 'r')
#    raw_gt = np.array(file_gt['im'])
#    print(raw_gt[0:1, 0:100, 50:51])
#     raw_gt = raw_gt[:15,:292,:154] # to test if stack is smaller than 20 z slices
#    raw_gt = raw_gt.astype('uint8')
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
#        raw_gt = np.vstack((raw_gt, pad))
        # create a mask to record this padding
        raw_mask = np.vstack((pad_mask, pad2))
    else:
        raw_mask = np.ones((raw_data.shape[0], raw_data.shape[1], raw_data.shape[2]), dtype='uint8')

#     print(raw_data.shape, raw_gt.shape, raw_mask.shape)
#     print(pad2.shape)
#     print(raw_mask[14:18,0:5,0:5])
#     print(raw_data.shape[0])

    # print(raw_data.shape[2])
#     print(output_dir)

    f = zarr.open(output_dir+'/'+zarr_name, 'w')
    f['raw'] = raw_data.astype('float32')
    f['raw'].attrs['resolution'] = (5,1,1)
    f['mask'] = raw_mask.astype('uint8')
    f['mask'].attrs['resolution'] = (5,1,1)

    
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "make Zarr directories for vessNet prediction")
	parser.add_argument('input_fname', type = str)
	parser.add_argument('--output_dir', type = str,default='.')
	args = parser.parse_args()
	makeZarr(args.input_fname, args.output_dir)










