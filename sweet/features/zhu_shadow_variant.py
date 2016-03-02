"""
Really simple implementations of the shadow-variant feature types in Zhu et al.'s 2010 paper
"Learning to Recognise Shadows in Monochromatic Natural Images"
"""

import cv2
import numpy as np


# TODO: this module is not ready to use yet


def smoothness(image, gaussian_radii=[3, 5]):
    def pair(ls):
        for i in range(0, len(ls) - 1):
            yield (ls[i], ls[i + 1])
    gblur = cv2.GaussianBlur(image, (5, 5), 0, 0)
    res = image - gblur
    return res

def local_max(image, r=3):
    rows, cols = image.shape[0:2]
    lm = np.zeros((rows, cols))
    for row in range(r, (rows - r)):
        for col in range(r, (cols - r)):
            patch = image[(row - r):(row + r),
                          (col - r):(col + r)]
            lm[row, col] = patch.max()
    return lm
