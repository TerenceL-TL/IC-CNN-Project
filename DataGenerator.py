import os
import time
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import tensorflow as tf
from Augmentation import *
# from test import show_predictions



class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, base_dir, list_IDs, batch_size=32, dim=(240,240), n_channels=3,
                 n_classes=2, shuffle=True, aug_dict = None):
        'Initialization'
        self.base_dir = base_dir
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.aug_dict = aug_dict
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, *self.dim), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # Add data augmentation here
            img = np.load(os.path.join(self.base_dir, 'image', ID))
            msk = np.load(os.path.join(self.base_dir, 'masks', ID))
            img, msk = augment_data(img, msk, self.aug_dict)
            # print(X.shape)
            # print(img.shape)
            X[i,...] = img
            y[i,...] = msk
        return X, y