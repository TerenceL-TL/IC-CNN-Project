
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import random

import torch
from matplotlib.widgets import Slider

def nii_to_npy(nii_file, npy_file):
    nii_img = nib.load(nii_file)
    
    nii_data = nii_img.get_fdata()

    num_slices = nii_data.shape[2]
    
    # Save each slice as a separate .npy file
    for i in range(num_slices):
        slice_data = nii_data[:, :, i]
        npy_file_slice = npy_file + f'_slice_{i}.npy'
        # print(npy_file_slice)
        np.save(npy_file_slice, slice_data)
    return()

# Data Visualization
# Choose set on the selection bar, then use the trackbar for moving up and down

# path related
img_path = 'dataset_segmentation/'
train_path = os.path.join(img_path, "train")

dirs = os.listdir(train_path)
random.shuffle(dirs)

num = len(dirs)

for (n, file_name) in enumerate(dirs):
    if n > 0.8 * num:
        nii_file_pla = os.path.join(train_path, file_name, file_name + "_fla.nii.gz")
        nii_file_seg = os.path.join(train_path, file_name, file_name + "_seg.nii.gz")
        nii_to_npy(nii_file_pla, os.path.join('Train', 'Image', file_name))
        nii_to_npy(nii_file_seg, os.path.join('Train', 'Target', file_name))
    else:
        nii_file_pla = os.path.join(train_path, file_name, file_name + "_fla.nii.gz")
        nii_file_seg = os.path.join(train_path, file_name, file_name + "_seg.nii.gz")
        nii_to_npy(nii_file_pla, os.path.join('Val', 'Image', file_name))
        nii_to_npy(nii_file_seg, os.path.join('Val', 'Target', file_name))
