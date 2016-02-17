#! /usr/bin/env python2

import importlib
import os
import sys

import cv2
import yaml


def imshow(*images):
    """
    Mostly used for debugging at the moment, but will probably display some nice output eventually.
    """
    import matplotlib.pyplot as plt

    # create the figure
    fig = plt.figure(figsize=(8, 8))

    for i in range(0, len(images)):
        # display original image with locations of patchea
        ax = fig.add_subplot(len(images), 1, i + 1)
        print(images[i].shape)
        ax.imshow(images[i], cmap=plt.cm.gray, interpolation="nearest", vmin=0, vmax=255)

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
                                *config[package][key]["parameters"]))
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
    data = {
        "ground_truth": [{"name": sys.argv[1], "img": cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)}],
        "images": [{"name": sys.argv[2], "img": cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)}]
    }

    # preprocess images (grayscale conversion, gaussian blur, etc)
    data["ground_truth"] = preprocess(config, data["ground_truth"])
    data["images"] = preprocess(config, data["images"])

    # extract features and evaluate
    features = {
        "ground_truth": extract_features(config, data["ground_truth"]),
        "images": extract_features(config, data["images"])
    }

    # compare!
    # TODO: write way of comparing computed features and ground truth
    for i in range(0, len(features["ground_truth"])):
        feature_type = features["ground_truth"][i]["type"]
        print(feature_type)

        for j in range(0, len(features["ground_truth"][i]["features"])):
            ground_truth = features["ground_truth"][i]["features"][j]
            image = features["images"][i]["features"][j]

            mse = ((ground_truth["image_features"] - image["image_features"]) ** 2).mean(axis=None)
            print("Frequency: %s, Angle: %s = Error: %f" % (image["parameters"][0], image["parameters"][1], mse))
