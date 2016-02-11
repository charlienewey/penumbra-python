#! /usr/bin/env python2

# Adapted from:
# http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html?highlight=texture


import math
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_filter


PATCH_SIZE = 80

def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

# read images as greyscale
image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
image_2 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)

loc = (136, 237)
patch_1 = image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE]
patch_2 = image_2[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE]

kerns_1 = []
kerns_2 = []

# append "real" part of array
def gfri(gf):
    return (np.real(gf[0]) + np.real(gf[1]))

for theta in [0, 45, 135]:
    gf = gabor_filter(patch_1, 0.2, theta=theta)
    kerns_1.append((theta, gfri(gf)))
    gf = gabor_filter(patch_2, 0.2, theta=theta)
    kerns_2.append((theta, gfri(gf)))

# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patchea
ax = fig.add_subplot(4, 2, 1)
ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
ax.plot(loc[1] + PATCH_SIZE / 2, loc[0] + PATCH_SIZE / 2, 'gs')
ax.set_xlabel('Shadow')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

ax = fig.add_subplot(4, 2, 2)
ax.imshow(image_2, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
ax.plot(loc[1] + PATCH_SIZE / 2, loc[0] + PATCH_SIZE / 2, 'gs')
ax.set_xlabel('Non-shadow')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# display the image patches
patches = [patch_1, patch_2]
for i, patch in enumerate(patches):
    ax = fig.add_subplot(4, len(patches), i + 3)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.set_xlabel('Patch %s' % i)

# display Gabor kernels
kernels = [kerns_1, kerns_2]
for i, kerns in enumerate(kernels):
    for j, (theta, kern) in enumerate(kerns):
        ax = fig.add_subplot(4, 3, 7 + (3 * i) + j)
        ax.imshow(kern, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
        ax.set_xlabel('Patch: %d, Theta: %d' % (i, theta))

# display the patches and plot
plt.show()
