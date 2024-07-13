import os
import numpy as np
import pandas as pd
import glob
import random
import matplotlib.pyplot as plt

from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from scipy.ndimage import rotate, zoom

def flip_horizontal(image, mask):
    return np.fliplr(image), np.fliplr(mask)

def flip_vertical(image, mask):
    return np.flipud(image), np.flipud(mask)

def rotate_image(image, mask, angle):
    rotated_img = rotate(image, angle, reshape=False)
    rotated_mask = rotate(mask, angle, reshape=False)
    return rotated_img, rotated_mask

def shift_image_and_mask(image, mask, shift_x, shift_y):
    
    assert image.shape == mask.shape, "Image and mask must have the same shape"
    
    h, w = image.shape[:2]
    
    # Create empty arrays for the shifted image and mask
    shifted_image = np.zeros_like(image)
    shifted_mask = np.zeros_like(mask)
    
    # Define the ranges for the original and shifted image
    orig_x_min = max(0, -shift_x)
    orig_x_max = min(w, w - shift_x)
    orig_y_min = max(0, -shift_y)
    orig_y_max = min(h, h - shift_y)
    
    shifted_x_min = max(0, shift_x)
    shifted_x_max = min(w, w + shift_x)
    shifted_y_min = max(0, shift_y)
    shifted_y_max = min(h, h + shift_y)
    
    # Perform the shift
    shifted_image[shifted_y_min:shifted_y_max, shifted_x_min:shifted_x_max] = \
        image[orig_y_min:orig_y_max, orig_x_min:orig_x_max]
    
    shifted_mask[shifted_y_min:shifted_y_max, shifted_x_min:shifted_x_max] = \
        mask[orig_y_min:orig_y_max, orig_x_min:orig_x_max]
    
    return shifted_image, shifted_mask

def augment_data(img, msk, aug_dict):
    if aug_dict is not None:
    # for img, msk in zip(images, masks):
        # Apply horizontal flip
        if aug_dict.get('horizontal_flip', False) and np.random.rand() > 0.5:
            img, msk = flip_horizontal(img, msk)

        # Apply vertical flip
        if aug_dict.get('vertical_flip', False) and np.random.rand() > 0.5:
            img, msk = flip_vertical(img, msk)

        # Apply rotation
        if aug_dict.get('rotation_range', 0) > 0:
            angle = np.random.uniform(-aug_dict['rotation_range'], aug_dict['rotation_range'])
            img, msk = rotate_image(img, msk, angle)

        # Apply Shift
        if (aug_dict.get('width_shift_range', 0) > 0) & (aug_dict.get('height_shift_range', 0) > 0):
            shift_w = int(np.random.uniform(-aug_dict['width_shift_range'], aug_dict['width_shift_range']) * img.shape[0])
            shift_h = int(np.random.uniform(-aug_dict['height_shift_range'], aug_dict['height_shift_range']) * img.shape[0])
            img, msk = shift_image_and_mask(img, msk, shift_w, shift_h)
    return [img, msk] # return 3
    # print(f'Augmented Image shape: {img.shape}, Augmented Mask shape: {msk.shape}')
