import random
import DataGenerator
from models import *
from function import *
# from data_gen import *
import os
from matplotlib.pyplot import *
from keras.utils import custom_object_scope
from keras.optimizers import Adam, Adamax
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# model = unet()
# model.compile(Adamax(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', dice_coef, iou_coef])

with custom_object_scope({'iou_coef': iou_coef, 'dice_coef': dice_coef}):
    model = load_model("final_model.h5")

model.summary()

epochs = 50
batch_size = 32
callbacks = [ModelCheckpoint('unet.hdf5', verbose=1, save_best_only=True)]

train_pla_dir = os.path.join('Train/3', 'image')
valid_pla_dir = os.path.join('Val/3', 'image')

train_list = os.listdir(train_pla_dir)
valid_list = os.listdir(valid_pla_dir)

train_dir = 'Train/3'
valid_dir = 'Val/3'

tr_aug_dict = dict(rotation_range=180.0,
                    width_shift_range=0.5,
                    height_shift_range=0.5,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

train_generator = DataGenerator.DataGenerator(train_dir, train_list, batch_size=batch_size, aug_dict=tr_aug_dict)
validation_generator = DataGenerator.DataGenerator(valid_dir, valid_list, batch_size=batch_size)
img_size = (240,240)

history = model.fit(train_generator,
                    epochs=epochs,
                    # verbose=1,
                    callbacks=callbacks,
                    validation_data=validation_generator,
                    )

model.save("final_model.h5")

# predicts = model.predict(test_gen, steps=len(test_df) / batch_size, verbose=1)

# for i in range(batch_size):
#     plt.imshow(predicts[i])
#     plt.show()


