import os
import numpy as np
import pandas as pd
import glob
import random
import matplotlib.pyplot as plt

from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

def create_df(data_dir):
    images_paths = glob.glob(f'{data_dir}image/*.npy')
    masks_paths = glob.glob(f'{data_dir}masks/*.npy')

    df = pd.DataFrame(data= {'images_paths': images_paths, 'masks_paths': masks_paths})

    return df

def split_df(df):
    train_df, dummy_df = train_test_split(df, train_size=0.8)
    valid_df, test_df = train_test_split(dummy_df, train_size=0.5)
    return train_df, valid_df, test_df

def create_gens(df, aug_dict):
    img_size = (240, 240)
    batch_size = 16

    img_gen = ImageDataGenerator(**aug_dict)
    msk_gen = ImageDataGenerator(**aug_dict)

    while True:
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]

            images = []
            masks = []

            for _, row in batch_df.iterrows():
                img = np.load(row['images_paths'])
                msk = np.load(row['masks_paths'])

                # Resize if needed
                img = resize(img, img_size, anti_aliasing=True)
                msk = resize(msk, img_size, anti_aliasing=True)

                # Add channel dimension
                img = np.expand_dims(img, axis=-1)
                msk = np.expand_dims(msk, axis=-1)

                images.append(img)
                masks.append(msk)

            images = np.array(images)
            masks = np.array(masks)

            # Normalize images and masks
            images = images / 255.0
            masks = masks / 255.0
            masks[masks > 0.5] = 1
            masks[masks <= 0.5] = 0

            yield (images, masks)

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
