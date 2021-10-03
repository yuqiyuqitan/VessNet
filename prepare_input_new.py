from PIL import Image
import numpy as np
from skimage import exposure
from skimage import io
import h5py
import os
import argparse

#load tiff file or czi file to python
def prepare_input(input_fname, output_fname, output_dir="."):
	"""
	"""
	im = io.imread(input_fname)

	#remove the top and bottom percent of the pixel, NOT ANYMORE
	#normalize the data from 0 to 1
	#p_min, p_max = np.percentile(im, (1,98))
	im_new = exposure.rescale_intensity(im, 
		out_range = (0,1))
	im_new = im_new.astype('float32')


	#save as h5 formate 
	save_path = os.path.join(output_dir, output_fname) + '.hdf5'
	hf = h5py.File(save_path, 'a')
	dset = hf.create_dataset('im', data=im_new, dtype='uint8')
	hf.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "prepare data for deepVess")
	parser.add_argument('input_fname', type = str)
	parser.add_argument('output_fname', type = str)
	parser.add_argument('--output_dir', type = str,default='.')
	args = parser.parse_args()
	prepare_input(args.input_fname, args.output_fname, args.output_dir)
