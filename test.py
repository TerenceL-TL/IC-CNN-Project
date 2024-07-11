from keras import layers
from keras.layers import *
from keras.models import *
# from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.models import load_model
from keras.utils import custom_object_scope
from models import *
from lost import *
from data_gen import *

model_path = "model_114514.h5" 
with custom_object_scope({'iou_coef': iou_coef, 'dice_coef': dice_coef}):
    model = load_model(model_path)

data_dir = 'Train/'
batch_size = 32
train_gen, valid_gen, test_gen, train_df, valid_df, test_df = init_data(data_dir)

predictions = model.predict(test_gen, steps=len(test_df) / batch_size, verbose=1)

# Assuming you have a batch size of 32 and want to inspect the first batch of predictions
for i in range(batch_size):
    # Visualize the predictions
    prediction = predictions[i]
    # Process the prediction as needed (e.g., thresholding, decoding, etc.)

    binary_mask = (prediction > 0.5).astype(np.uint8)
    img = np.array(binary_mask)

    # Display the binary mask
    plt.imshow(img)
    plt.title(f'Prediction {i + 1}')
    plt.show()