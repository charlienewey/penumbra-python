import os
import sys

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage import img_as_ubyte

from tqdm import tqdm

import sweet.features.zhu_shadow_variant

def imshow(*images):
    """
    Display an image in a window. Mostly used for debugging at the moment.
    """
    import matplotlib.pyplot as plt

    # create the figure
    fig = plt.figure(figsize=(8, 8))

    for i in range(0, len(images)):
        # display original image with locations of patchea
        ax = fig.add_subplot(len(images), 1, i + 1)
        ax.imshow(images[i], cmap=plt.cm.gray, interpolation="nearest", vmin=0, vmax=255)

    # display the patches and plot
    plt.show()



if __name__ == "__main__":
    path = os.path.abspath(os.path.join(os.getcwd(), sys.argv[1]))
    iv = cv2.VideoCapture(path)
    fc = int(iv.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    fourcc = cv2.cv.FOURCC(*"XVID")
    ov = cv2.VideoWriter(sys.argv[2], fourcc, 20.0, (1920, 1080))

    for i in tqdm(range(0, fc)):
        # read next frame
        ret, frame = iv.read()

        # segment and threshold
        lmx = ndi.maximum_filter(frame, size=10, mode='constant')
        thresh = threshold_otsu(lmx, 3)
        threshed = lmx >= (thresh - 25)

        # morphological stuff
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        detection = cv2.morphologyEx(img_as_ubyte(threshed), cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        detection = cv2.morphologyEx(img_as_ubyte(detection), cv2.MORPH_CLOSE, kernel)

        # write the detection
        ov.write(detection)
