import numpy as np

def msk_thresh(msk, threshold):
    return (msk > threshold).astype(np.uint8)