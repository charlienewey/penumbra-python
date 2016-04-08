from skimage.filter import gaussian_filter as _gaussian

def gaussian(img, sigma=8):
    return _gaussian(img, sigma=sigma)
