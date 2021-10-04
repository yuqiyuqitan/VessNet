import re
import neuroglancer
import numpy as np
import h5py
import os
from numpy.core.shape_base import vstack
import zarr
import sys
import gunpowder as gp

resolution = [5,1,1]
offset = [0,]*3

f = zarr.open(sys.argv[1])

viewer = neuroglancer.Viewer()

with viewer.txn() as s:

    for ds_name in ['raw_aug','gt_aug']:

        data = f[ds_name][:]
        offset = f[ds_name].attrs['offset']
        resolution = f[ds_name].attrs['resolution']

        dimensions = neuroglancer.CoordinateSpace(
            names=['b^','z','y','x'],
            units='nm',
            scales=[1,]+resolution
        )
        
        vol = neuroglancer.LocalVolume(
            data=data,
            voxel_offset=[0,]+offset,
            dimensions=dimensions
        )

        s.layers[ds_name] = neuroglancer.ImageLayer(source=vol)

print(viewer)

        




