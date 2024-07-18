from matplotlib import pyplot as plt
from function import *
from models import *

from keras import layers
from keras.layers import *
from keras.models import *
from keras.models import *
from keras.utils import custom_object_scope

import plotly.graph_objects as go

import numpy as np
import nibabel as nib

import os

from postprocess import msk_thresh

thresh = 0.3

# path related
img_path = 'dataset_segmentation/'
train_path = os.path.join(img_path, "test_fla")
model_path = "final_model.h5"

with custom_object_scope({'iou_coef': iou_coef, 'dice_coef': dice_coef}):
    model = load_model(model_path)

dirs = os.listdir(train_path)
dirs.sort()

num = len(dirs)

def process(read_path, save_path):
    fla_data = nib.load(read_path).get_fdata().astype(np.uint8)

    num_slices = fla_data.shape[2]

    X = np.zeros((fla_data.shape[2], fla_data.shape[0], fla_data.shape[1]))

    seg_data = np.zeros_like(fla_data)

    for i in range(num_slices):
        slice_data_pla = fla_data[:, :, i]
        X[i,] = slice_data_pla

    prediction = model.predict(X)

    for i in range(prediction.shape[0]):
        slice_data_pred = prediction[i]
        seg_data[:, :, i] = msk_thresh(slice_data_pred[:, :, 0], thresh)
    print(seg_data.shape)
    nifti_image = nib.Nifti1Image(seg_data, affine=np.eye(4))
    nib.save(nifti_image, save_path)

for (n, file_name) in enumerate(dirs):
    nii_file_pla = os.path.join(train_path, file_name, file_name + "_fla.nii.gz")
    nii_file_seg = os.path.join(train_path, file_name, file_name + "_seg.nii.gz")
    process(nii_file_pla, nii_file_seg)




