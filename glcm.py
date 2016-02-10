#! /usr/bin/env python2

# Adapted from:
# http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html?highlight=texture

import random
import sys

import cv2
import matplotlib.pyplot as plt
import skimage.io
from skimage.feature import greycomatrix, greycoprops
from skimage import data


PATCH_SIZE = 7

# read image as greyscale
image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
image_2 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)

locations = [(136, 237)]

patches = []
for loc in locations:
    patches.append(image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])
    patches.append(image_2[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch
xs = []
ys = []
for patch in patches:
    glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])

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

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 1, 3)
ax.plot(xs[:len(patches)], ys[:len(patches)], 'go',
        label='Shadow vs non-shadow patch')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLVM Correlation')
ax.legend()

# display the image patches
for i, patch in enumerate(patches):
    ax = fig.add_subplot(3, len(patches), len(patches)*1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)

# display the patches and plot
fig.suptitle('GLCM feature comparison - shadow/non-shadow area', fontsize=14)
plt.show()
