#! /usr/bin/env python2

import importlib
import os
import sys

import cv2
import yaml


def read_config(path):
    with open(CONFIG_FILE) as conf_in:
        config = yaml.load(conf_in)
    return config

def preprocess(config, images):
    # import and run preprocessing stage
    package = "preprocess"
    if package in config:
        print(package)
        for key, value in config[package].items():
            module = ".".join([package, key])
            mod = importlib.import_module(module)
            for i in range(0, len(images)):
                print("%s: %s" % (config[package][key]["desc"], images[i]["name"]))
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
                features.append(mod.extract_features(images[i]["img"]))
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
    in_1 = {"name": sys.argv[1], "img": cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)}
    in_2 = {"name": sys.argv[2], "img": cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)}
    assert in_1 is not None and in_2 is not None
    images = [in_1, in_2]

    # preprocess images (grayscale conversion, gaussian blur, etc)
    images = preprocess(config, images)

    # extract features and evaluate
    features = extract_features(config, images)
