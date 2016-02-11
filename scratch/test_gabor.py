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
from skimage.filters import gabor_kernel


PATCH_SIZE = 80

def compute_feats(image, kernel):
    feats = np.zeros((2), dtype=np.double)
    filtered = ndi.convolve(image, kernel, mode='wrap')
    feats[0] = filtered.mean()
    feats[1] = filtered.var()
    return feats

def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

# read images as greyscale
image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
image_2 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)

locations = [(136, 237)]
patches = []
for loc in locations:
    patches.append(image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])
    patches.append(image_2[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])

ckernels = []
kern = gabor_kernel(0.4)
for patch in patches:
    ckernels.append(power(patch, kern))

# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patchea
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
for (y, x) in locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
ax.set_xlabel('Shadow')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

ax = fig.add_subplot(3, 2, 2)
ax.imshow(image_2, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
for (y, x) in locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
ax.set_xlabel('Non-shadow')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# display the image patches
for i, patch in enumerate(patches):
    ax = fig.add_subplot(3, len(patches), i + 3)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.set_xlabel('Patch %s' % i)

# display Gabor kernels
for i, kernel in enumerate(ckernels):
    ax = fig.add_subplot(3, 2, i + 5)
    ax.imshow(np.real(kernel), cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
    ax.set_xlabel('Patch %s' % i)


# Check that the kernels aren't exactly the same
a, b = (ckernels[0], ckernels[1])
if np.equal(a, b).all():
    print("Something went wrong")

# display the patches and plot
plt.show()
