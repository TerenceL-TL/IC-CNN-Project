from keras import layers
from keras.layers import *
from keras.models import *
from keras.models import *
from keras.utils import custom_object_scope
from models import *

from DataGenerator import *
from function import *
from Augmentation import *
# from train import model

batch_size = 32

valid_pla_dir = os.path.join('Val', 'image')

valid_list = os.listdir(valid_pla_dir)

valid_dir = 'Val/'

model_path = "model_right_3.h5" 

with custom_object_scope({'iou_coef': iou_coef, 'dice_coef': dice_coef}):
    model = load_model(model_path)

validation_generator = DataGenerator(valid_dir, valid_list, batch_size=batch_size)

test_sets, test_masks = validation_generator.__getitem__(1)

preds = model.predict(test_sets)

for i in range(batch_size):
    test_imag = test_sets[i]
    test_mask = test_masks[i]
    pred_mask = preds[i]

    plt.figure()

    plt.subplot(1, 3, 1)
    plt.imshow(test_imag)
    plt.title("test_imag")

    plt.subplot(1, 3, 2)
    plt.imshow(test_mask)
    plt.title("test_mask")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask)
    plt.title("pred_mask")

    plt.show()