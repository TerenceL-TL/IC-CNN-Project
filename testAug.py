from Augmentation import *
from PIL import Image

img_path = "Train/image/001_slice_31.png"
seg_path = "Train/masks/001_slice_31.png"

img_png = Image.open(img_path)
seg_png = Image.open(seg_path)

img = np.array(img_png)
seg = np.array(seg_png)

# img, seg = rotate_image(img, seg, 20)

print(img.shape)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(seg)

plt.show()