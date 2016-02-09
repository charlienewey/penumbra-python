#! /usr/bin/env python2

import sys

import skimage
import skimage.io
import skimage.feature
import skimage.viewer


path = sys.argv[1]

im = skimage.io.imread(path, as_grey=True)
x = skimage.feature.greycoprops(im, {"prop": "correlation"})

viewer = skimage.viewer.ImageViewer(x)
viewer.show()
