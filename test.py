import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib

save = "001_seg.jpg"
ppath = "001_seg.nii.gz"

data = nib.load(ppath).get_fdata()

slicesss = data.shape[2]
print(slicesss)

for i in range(slicesss):
    image = data[:,:,i]
    plt.imshow(image)
    plt.show()
    print(image.shape)
    image =  image.astype(np.uint8)
    img = Image.fromarray(image)

    plt.imshow(img)
    plt.show()
    img.save(save)
