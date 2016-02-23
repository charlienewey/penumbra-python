import math

import skimage.filters


def variance_difference(image_1, image_2):
    def _var_dif(img_1, img_2):
        return math.sqrt((img_1.var() - img_2.var()) ** 2)

    if isinstance(image_1, list):
        var_dif = 0
        for i in range(0, len(image_1)):
            var_dif += _var_dif(image_1[i], image_2[i])
        return var_dif / len(image_1)
    else:
        return _var_dif(image_1, image_2)


def mean_squared_error(image_1, image_2):
    def _mse(img_1, img_2):
        return ((img_1 - img_2) ** 2).mean(axis=None)

    if isinstance(image_1, list):
        err = 0
        for i in range(0, len(image_1)):
            err += _mse(image_1[i], image_2[i])
        return (err / len(image_1))
    else:
        return _mse(image_1, image_2)


def gabor_filter(image, frequency, theta):
    if isinstance(image, list):
        filters = []
        for i in range(0, len(image)):
            filters.append(skimage.filters.gabor_filter(image[i], frequency, theta)[0])
        return filters
    else:
        return skimage.filters.gabor_filter(image, frequency, theta)[0]
