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

def augment_data(images, masks, aug_dict):

    for img, msk in zip(images, masks):
        yield [img, msk]
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
            yield [img, msk]
            img, msk = rotate_image(img, msk, angle)
            yield [img, msk]
            img, msk = rotate_image(img, msk, angle)
            yield [img, msk] # return 3
            print(f'Augmented Image shape: {img.shape}, Augmented Mask shape: {msk.shape}')

def create_df(data_dir):
    images_paths = glob.glob(f'{data_dir}image/*.png')
    masks_paths = glob.glob(f'{data_dir}masks/*.png')

    df = pd.DataFrame(data= {'images_paths': images_paths, 'masks_paths': masks_paths})

    return df

def split_df(df):
    train_df, dummy_df = train_test_split(df, train_size=0.8)
    valid_df, test_df = train_test_split(dummy_df, train_size=0.5)
    return train_df, valid_df, test_df

def normal_gens(df):
    image_paths = df['images_paths']
    mask_paths = df['masks_paths']

    for (img_path, mask_path) in zip(image_paths, mask_paths):
        train_image = np.load(img_path)
        train_mask = np.load(mask_path)
        
        # Reshape if necessary to ensure the image has the shape (height, width, channels)
        if len(train_image.shape) == 2:
            train_image = np.expand_dims(train_image, axis=-1)
        if len(train_mask.shape) == 2:
            train_mask = np.expand_dims(train_mask, axis=-1)
        
        yield [train_image, train_mask]

    

def create_gens(df, aug_dict):
    img_size = (240, 240)
    batch_size = 16

    img_gen = ImageDataGenerator(**aug_dict)
    msk_gen = ImageDataGenerator(**aug_dict)

    # Create general generator
    image_gen = img_gen.flow_from_dataframe(df, x_col='images_paths', class_mode=None, color_mode='grayscale', target_size=img_size,
                                            batch_size=batch_size, save_to_dir=None, save_prefix='image', seed=1)

    mask_gen = msk_gen.flow_from_dataframe(df, x_col='masks_paths', class_mode=None, color_mode='grayscale', target_size=img_size,
                                            batch_size=batch_size, save_to_dir=None, save_prefix= 'mask', seed=1)

    gen = zip(image_gen, mask_gen)
    # Normalize images and masks
    for (img_, msk_) in gen:
        # img_ = resize(img_, img_size, anti_aliasing=True)
        # msk_ = resize(msk_, img_size, anti_aliasing=True)
        # img_ = np.expand_dims(img_, axis=-1)
        # msk_ = np.expand_dims(msk_, axis=-1)
        img_ = img_ / 255
        msk_ = msk_ / 255
        msk_[msk_ > 0.5] = 1
        msk_[msk_ <= 0.5] = 0

        yield (img_, msk_)

def show_images(images, masks):
    plt.figure(figsize=(12, 12))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        img_path = images[i]
        mask_path = masks[i]

        image = np.load(img_path)
        mask = np.load(mask_path)

        plt.imshow(image.squeeze(), cmap='gray')
        plt.imshow(mask.squeeze(), cmap='jet', alpha=0.4)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def init_data(data_dir):
    df = create_df(data_dir)
    train_df, valid_df, test_df = split_df(df)

    tr_aug_dict = dict(rotation_range=0.2,
                       width_shift_range=0.05,
                       height_shift_range=0.05,
                       shear_range=0.05,
                       zoom_range=0.05,
                       horizontal_flip=True,
                       fill_mode='nearest')

    train_gen = create_gens(train_df, aug_dict=tr_aug_dict)
    valid_gen = create_gens(valid_df, aug_dict={})
    test_gen = create_gens(test_df, aug_dict={})
    return train_gen, valid_gen, test_gen, train_df, valid_df, test_df

train_gen, valid_gen, test_gen, train_df, valid_df, test_df = init_data("Train/")
