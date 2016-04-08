from skimage.util import import view_as_blocks


def to_patches(img, patch_size=20):
    """
    Assuming a 2-dimensional image, divide into patch_size * patch_size pixel patches.
    """
    return view_as_blocks(img, (patch_size, patch_size))
