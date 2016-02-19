import numpy as np

from skimage.feature import local_binary_pattern


def mean_squared_error(image_1, image_2):
    return ((np.asarray(image_1) - np.asarray(image_2)) ** 2).mean(axis=None)


def lbp(image, num_points, radius, method):
    return local_binary_pattern(image, num_points, radius, method)
