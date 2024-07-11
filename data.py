
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import random

import torch

from PIL import Image as IM
from matplotlib.widgets import Slider

def nii_to_jpg(nii_file, jpg_file):
    nii_img = nib.load(nii_file)
    
    nii_data = nii_img.get_fdata()

    num_slices = nii_data.shape[2]
    
    # Save each slice as a separate .npy file
    for i in range(num_slices):
        slice_data = nii_data[:, :, i]
        jpg_file_slice = jpg_file + f'_slice_{i}.jpg'
        slice_data =  slice_data.astype(np.uint8)
        img2save = IM.fromarray(slice_data)
        img2save.save(jpg_file_slice)
    return()

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
        nii_to_jpg(nii_file_pla, os.path.join('Train', 'Image', file_name))
        nii_to_jpg(nii_file_seg, os.path.join('Train', 'Target', file_name))
    else:
        nii_file_pla = os.path.join(train_path, file_name, file_name + "_fla.nii.gz")
        nii_file_seg = os.path.join(train_path, file_name, file_name + "_seg.nii.gz")
        nii_to_jpg(nii_file_pla, os.path.join('Val', 'Image', file_name))
        nii_to_jpg(nii_file_seg, os.path.join('Val', 'Target', file_name))
