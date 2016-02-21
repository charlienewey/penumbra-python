import numpy as np

from skimage.feature import greycomatrix
from skimage.feature import greycoprops


def variance_difference(image_1, image_2):
    var_1 = greycoprops(image_1, prop="contrast")
    var_2 = greycoprops(image_2, prop="contrast")
    v = np.sqrt((np.ravel(var_1) - np.ravel(var_2)) ** 2)
    return sum(v) / len(v)


def mean_squared_error(image_1, image_2):
    return ((np.asarray(image_1) - np.asarray(image_2)) ** 2).mean(axis=None)


def glcm(image, angles, dists):
    """
    Extract features from the image.

    Args:
        image:  An image read in by cv2.imread(...)
        angles: A list containing angles, e.g. [0, 45, 90, 135]
        dists:  A list containing pixel distances, e.g. [0, 1, 2, 3]
    Returns:
        A gray-level co-occurrence matrix.
    """
    return greycomatrix(image, angles, dists, 256, symmetric=True, normed=True)
