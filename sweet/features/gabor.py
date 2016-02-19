import skimage.filters


def mean_squared_error(image_1, image_2):
    return ((image_1 - image_2) ** 2).mean(axis=None)


def correlation(ground_truth, features):
    pass


def gabor_filter(image, frequency, theta):
    return skimage.filters.gabor_filter(image, frequency, theta)[0]
