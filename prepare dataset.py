#prepare dataset

import zarr
import gunpowder as gp
path = "/mnt/efs/woods_hole/danceParty/tiff_images/02_h5/"
fname_raw = 'image/01_HaftJavaherian_DeepVess2018_training_1.hdf5'
fname_gt = 'label/01_HaftJavaherian_DeepVess2018_training_1.hdf5'
zarr_name = 'sample_data.zarr'
#read in data from h5 file
file_raw = h5py.File(os.path.join(path,fname_raw), 'r')
raw_data = np.array(file_raw['im']) #is this z, y, x?
file_gt = h5py.File(os.path.join(path,fname_gt), 'r')
raw_gt = np.array(file_gt['im'])

#store image in zarr container
f = zarr.open(zarr_name, 'w')
f['raw'] = raw_data
f['raw'].attrs['resolution'] = (5,1,1)
f['ground_truth'] = raw_gt
f['ground_truth'].attrs['resolution'] = (5,1,1)


#padding if the data z < 20, add padding on z #mask it in loss function


#random cropping along x y so that 50% train, 25% val, 25% test


#need to pad for z for some images
