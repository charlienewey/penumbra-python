from skimage.filters import gaussian as _gaussian

def gaussian(imgs, sigma=8):
    return [_gaussian(img, sigma=sigma) for img in imgs]
