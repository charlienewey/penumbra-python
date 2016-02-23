import numpy as np


def to_patches(imgs, patch_size=20):
    """
    Assuming a 2-dimensional image, divide into patch_size * patch_size pixel patches.
    """
    patches = []
    for img in imgs:
        img_patches = []
        rows, cols = img.shape[0:2]
        for row in range(0, rows, patch_size):
            for col in range(0, cols, patch_size):
                patch = img[row:row + patch_size,col:col + patch_size]
                img_patches.append(patch)
        patches.append(img_patches)
    return patches
