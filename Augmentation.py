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
    return [img, msk] # return 3
    # print(f'Augmented Image shape: {img.shape}, Augmented Mask shape: {msk.shape}')

