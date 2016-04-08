from scipy import ndimage as ndi


def local_max(img, neighbourhood=3):
    return ndi.maximum_filter(img, size=neighbourhood)
