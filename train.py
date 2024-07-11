from models import *
from lost import *
from data_gen import *
from matplotlib.pyplot import *
from keras.optimizers import Adam, Adamax
from keras.callbacks import EarlyStopping, ModelCheckpoint

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


model = unet()
# model = load_model('unet.hdf5')

model.compile(Adamax(learning_rate= 0.001), loss= 'binary_crossentropy', metrics= ['accuracy', iou_coef, dice_coef])
model.summary()

data_dir = 'Train/'
epochs = 10
batch_size = 16
callbacks = [ModelCheckpoint('unet.hdf5', verbose=0, save_best_only=True)]

train_gen, valid_gen, test_gen, train_df, valid_df, test_df = init_data(data_dir)

history = model.fit(train_gen,
                    steps_per_epoch=len(train_df) / batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data = valid_gen,
                    validation_steps=len(valid_df) / batch_size)

model.save("model_114514.h5")

