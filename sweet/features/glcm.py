import numpy as np

from skimage.feature import greycomatrix
from skimage.feature import greycoprops


def variance_difference(image_1, image_2):
    def _var_dif(img_1, img_2):
        var_1 = greycoprops(img_1, prop="contrast")
        var_2 = greycoprops(img_2, prop="contrast")
        v = np.sqrt((np.ravel(var_1) - np.ravel(var_2)) ** 2)
        return v.mean(axis=None)

    if isinstance(image_1, list):
        var_dif = 0
        for i in range(0, len(image_1)):
            return _var_dif(image_1[i], image_2[i])
        return var_dif / len(image_1)
    else:
        return _var_dif(image_1, image_2)

def mean_squared_error(image_1, image_2):
    if isinstance(image_1, list):
        err = 0
        for i in range(0, len(image_1)):
            err += ((image_1[i] - image_2[i]) ** 2).mean(axis=None)
        return (err / len(image_1))
    else:
        return ((image_1[i] - image_2[i]) ** 2).mean(axis=None)


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
    if isinstance(image, list):
        glcms = []
        for i in range(0, len(image)):
            glcms.append(greycomatrix(image[i], angles, dists, 256, symmetric=True, normed=True))
        return glcms
    else:
        return greycomatrix(image, angles, dists, 256, symmetric=True, normed=True)
