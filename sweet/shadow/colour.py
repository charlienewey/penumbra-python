import numpy as np

from skimage.filter import threshold_otsu


def variance_otsu(image, labels):
    """
    Read pixel intensities from 'image' for each class given in 'labels' (a corresponding cluster
    label array), and use Otsu's binarisation to threshold the image.

    :param image: Pre-processed input image
    :param labels: Array with same shape as input image, containing cluster labels
    """
    img = image.ravel()
    shadow_seg = img.copy()

    hist, _ = np.histogram(labels)
    n_clusters = hist.shape[0]
    for i in range(0, n_clusters):
        # set up mask of pixel indices matching cluster
        mask = np.nonzero((labels.ravel() == i) == True)[0]
        if len(mask) > 0:
            if img[mask].var() > 0.005:
                thresh = threshold_otsu(img[mask])
                shadow_seg[mask] = shadow_seg[mask] < thresh
            else:
                shadow_seg[mask] = 0

    shadow_seg = shadow_seg.reshape(*image.shape)
    return shadow_seg
