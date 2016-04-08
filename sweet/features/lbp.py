import math

import numpy as np

from skimage.feature import local_binary_pattern


def variance_difference(image_1, image_2):
    def _var_dif(img_1, img_2):
        return math.sqrt((img_1.var() - img_2.var()) ** 2)
    return _var_dif(image_1, image_2)


def mean_squared_error(image_1, image_2):
    def _mse(img_1, img_2):
        return ((img_1 - img_2) ** 2).mean(axis=None)
    return _mse(image_1, image_2)


def lbp(image, num_points, radius, method):
    return local_binary_pattern(image, num_points, radius, method)
