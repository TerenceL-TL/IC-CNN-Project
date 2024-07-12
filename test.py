from keras import layers
from keras.layers import *
from keras.models import *
# from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.models import load_model
from keras.utils import custom_object_scope
from models import *
from function import *
from Augmentation import *

model_path = "model_114514.h5" 
with custom_object_scope({'iou_coef': iou_coef, 'dice_coef': dice_coef}):
    model = load_model(model_path)

data_dir = 'Train/'
batch_size = 16
train_gen, valid_gen, test_gen, train_df, valid_df, test_df = init_data(data_dir)

predictions = model.predict(test_gen, steps=len(test_df) / batch_size, verbose=1)

for i in range(batch_size):
    plt.imshow(predictions[i])
    plt.show()