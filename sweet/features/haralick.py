import math

import numpy as np

import mahotas


def variance_difference(image_1, image_2):
    return math.sqrt((image_1.var() - image_2.var()) ** 2)


def mean_squared_error(image_1, image_2):
    image_1 = np.transpose(image_1)
    image_2 = np.transpose(image_2)

    e = ((image_1 - image_2) ** 2).mean()
    return math.sqrt(e)


def haralick(image):
    return mahotas.features.haralick(image)
