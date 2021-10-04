#prepare dataset
import re
import neuroglancer
import numpy as np
import h5py
import os
from numpy.core.shape_base import vstack
import zarr
import gunpowder as gp

path = "/mnt/efs/woods_hole/danceParty/tiff_images/02_h5/"
fname_raw = 'image/01_HaftJavaherian_DeepVess2018_training_1.hdf5'
fname_gt = 'label/01_HaftJavaherian_DeepVess2018_training_1.hdf5'
zarr_name = 'sample_data.zarr'
#read in data from h5 file
file_raw = h5py.File(os.path.join(path,fname_raw), 'r')
raw_data = np.array(file_raw['im'])
file_gt = h5py.File(os.path.join(path,fname_gt), 'r')
raw_gt = np.array(file_gt['im'])
raw_gt = raw_gt.astype('uint8')

# padding if the data z < 20, add padding on z 
if raw_data.shape[0] < 20:
    pad = np.zeros((20-raw_data.shape[0], raw_data.shape[1], raw_data.shape[2]))
    raw_data = np.vstack(raw_data, pad)
    raw_gt = np.vstack(raw_gt, pad)
    # create a mask to record this padding
    pad2 = np.zeros((20-raw_data.shape[0], raw_data.shape[1], raw_data.shape[2]), dtype='uint8')
    pad_mask = np.ones((raw_data.shape[0], raw_data.shape[1], raw_data.shape[2]), dtype='uint8')
    raw_mask = np.vstack(pad_mask, pad2)
else:
    raw_mask = np.ones((raw_data.shape[0], raw_data.shape[1], raw_data.shape[2]), dtype='uint8')
#splitby the x, y axis 50%, 25%, 25% into train, test, val

resolution = [5,1,1]
offset = [0,]*3

f = zarr.open(zarr_name, 'a')

for ds_name, data in [
    ('raw', raw_data),
    ('gt', raw_gt),
    ('mask', raw_mask)]:

    f[ds_name] = data
    f[ds_name].attrs['offset'] = offset
    f[ds_name].attrs['resolution'] = resolution
