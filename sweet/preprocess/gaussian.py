from skimage.filters import gaussian

def gaussian(imgs, sigma=8):
    return [gaussian(img, sigma=sigma) for img in imgs]
