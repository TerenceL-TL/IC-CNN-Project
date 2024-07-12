
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

def nii_to_npy(nii_pla_file, nii_seg_file, npy_file, idx):
    nii_img_pla = nib.load(nii_pla_file).get_fdata()
    nii_img_seg = nib.load(nii_seg_file).get_fdata()
    
    num_slices = nii_img_pla.shape[2]
    
    # Save each slice as a separate .npy file
    for i in range(num_slices):
        slice_data_pla = nii_img_pla[:, :, i]
        slice_data_seg = nii_img_seg[:, :, i]
        if np.all(slice_data_pla == 0):
            continue
        slice_data_seg *= 255

        slice_data_pla = slice_data_pla.astype(np.uint8)
        slice_data_seg = slice_data_seg.astype(np.uint8)

        pla_file_slice = os.path.join(npy_file, 'image', idx + f'_slice_{i}.png')
        seg_file_slice = os.path.join(npy_file, 'masks', idx + f'_slice_{i}.png')
        img_pla = IM.fromarray(slice_data_pla)
        img_seg = IM.fromarray(slice_data_seg)
        img_pla.save(pla_file_slice)
        img_seg.save(seg_file_slice)
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
    nii_to_npy(nii_file_pla, nii_file_seg, os.path.join('Train'), file_name)




