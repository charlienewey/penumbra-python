import math

import numpy as np

import mahotas


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
        img_1 = np.transpose(img_1)
        img_2 = np.transpose(img_2)
        e = ((img_1 - img_2) ** 2).mean()
        return math.sqrt(e)

    if isinstance(image_1, list):
        err = 0
        for i in range(0, len(image_1)):
            err += _mse(image_1[i], image_2[i])
        return (err / len(image_1))
    else:
        return _mse(image_1, image_2)


def haralick(image):
    if isinstance(image, list):
        haralick_feats = []
        for i in range(0, len(image)):
            haralick_feats.append(mahotas.features.haralick(image[i], num_points, radius, method)[0])
        return haralick_feats 
    else:
        return mahotas.features.haralick(image)
