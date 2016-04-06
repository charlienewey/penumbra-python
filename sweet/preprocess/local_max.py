from scipy import ndimage as ndi


def local_max(imgs, neighbourhood=3):
    return [ndi.maximum_filter(img, size=neighbourhood) for img in imgs]
