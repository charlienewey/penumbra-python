#! /usr/bin/env python2

import importlib
import os
import sys

import cv2
import yaml


def imshow(image):
    """
    Mostly used for debugging at the moment, but will probably display some nice output eventually.
    """
    import matplotlib.pyplot as plt

    # create the figure
    fig = plt.figure(figsize=(8, 8))

    # display original image with locations of patchea
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image, cmap=plt.cm.gray, interpolation="nearest", vmin=0, vmax=255)

    # display the patches and plot
    plt.show()


def read_config(path):
    with open(CONFIG_FILE) as conf_in:
        config = yaml.load(conf_in)
    return config


def preprocess(config, images):
    package = "preprocess"
    if package in config:
        print(package)
        for key, value in config[package].items():
            module = ".".join([package, key])
            mod = importlib.import_module(module)
            for i in range(0, len(images)):
                print("%s: %s" % (config[package][key]["name"], images[i]["name"]))
                images[i]["img"] = mod.preprocess(images[i]["img"])
    return images


def extract_features(config, images):
    features = []
    package = "features"
    if package in config:
        print(package)
        for key, value in config[package].items():
            module = ".".join([package, key])
            mod = importlib.import_module(module)

            for i in range(0, len(images)):
                print("%s: %s" % (config[package][key]["name"], images[i]["name"]))
                features.append(mod.features(images[i]["img"],
                                *config[package][key]["features_parameters"]))
    return features


if __name__ == "__main__":
    # set constants here - will later replace with command-line opts
    CONFIG_FILE = os.path.join(os.getcwd(), "config/config.yml")

    # append current directory to system path
    cur_dir = os.path.dirname(__file__)
    sys.path.append(cur_dir)

    # get configuration
    config = read_config(CONFIG_FILE)

    # read input files and make sure everything's hunky-dory
    # TODO: replace this with something a bit more sensible (list of files, etc)
    ground_truth = [{"name": sys.argv[1], "img": cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)}]
    data = [{"name": sys.argv[2], "img": cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)}]
    assert ground_truth is not None and data is not None

    # preprocess images (grayscale conversion, gaussian blur, etc)
    ground_truth = preprocess(config, ground_truth)
    data = preprocess(config, data)

    # extract features and evaluate
    features = extract_features(config, data)

    # compare!
    # TODO: write way of comparing computed features and ground truth
