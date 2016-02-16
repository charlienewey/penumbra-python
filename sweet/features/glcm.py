from skimage.feature import greycomatrix
from skimage.feature import greycoprops


def squared_error(ground_truth, features):
    raise Exception("Not implemented yet")


def correlation(ground_truth, features):
    raise Exception("Not implemented yet")


def features(image, angles, dists):
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
