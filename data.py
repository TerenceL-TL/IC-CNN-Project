
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import glob
import random

from PIL import Image as IM

from skimage.color import rgb2gray
from skimage.morphology import label
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow, concatenate_images

def nii_to_npy(nii_file, npy_file):
    nii_img = nib.load(nii_file)
    
    nii_data = nii_img.get_fdata()

    num_slices = nii_data.shape[2]
    
    # Save each slice as a separate .npy file
    for i in range(num_slices):
        slice_data = nii_data[:, :, i]
        npy_file_slice = npy_file + f'_slice_{i}.npy'
        # slice_data = slice_data.astype(np.uint8)
        # img = IM.fromarray(slice_data)
        # img.save(jpg_file_slice)
        np.save(npy_file_slice, slice_data)
    return()

# path related
img_path = 'dataset_segmentation/'
train_path = os.path.join(img_path, "train")

dirs = os.listdir(train_path)
random.shuffle(dirs)

num = len(dirs)

for (n, file_name) in enumerate(dirs):
    nii_file_pla = os.path.join(train_path, file_name, file_name + "_fla.nii.gz")
    nii_file_seg = os.path.join(train_path, file_name, file_name + "_seg.nii.gz")
    nii_to_npy(nii_file_pla, os.path.join('Train', 'image', file_name))
    nii_to_npy(nii_file_seg, os.path.join('Train', 'masks', file_name))




