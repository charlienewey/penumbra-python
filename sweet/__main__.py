#! /usr/bin/env python2

import importlib
import itertools
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
        ax.imshow(images[i], cmap=plt.cm.gray, interpolation="nearest", vmin=0, vmax=255)

    # display the patches and plot
    plt.show()


def read_config(path):
    with open(CONFIG_FILE) as conf_in:
        config = yaml.load(conf_in)
    return config


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
    ground_truths = [{"name": sys.argv[1], "img": cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)}]
    images = [{"name": sys.argv[2], "img": cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)}]
    assert len(ground_truths) == len(images)


    # This is probably a bit indecipherable if you aren't familiar with introspection, so... sorry
    # for that. This section imports all of the necessary modules and their functions, fetches the
    # functions, and appends them to a list, along with their parameters -- effectively creating a
    # processing pipeline. This pipeline can then be applied multiple times to different image
    # pairs, and lends itself well to parallelisation.
    #
    # TODO: chuck a few links on here so that peeps can read what on earth this code is on about
    prep_chain = []
    package = "preprocess"
    for stage in config[package]:
        # import module and fetch the specified method
        module = ".".join((package, stage["name"]))
        mod = importlib.import_module(module)
        method = getattr(mod, stage["method"])

        # fetch the parameters, generate all parameter possibilities and chuck them into a lambda
        # function so that you only need to call: "method(image)".
        if "parameters" in stage:
            for args in itertools.product(*stage["parameters"]):
                prep_chain.append({
                    "description": stage["description"],
                    "method": lambda im: method(im, *args),
                    "parameters": args
                })
        else:
            prep_chain.append({
                "description": stage["description"],
                "method": method
            })


    # This section is similar to the last, except the evaluation stage happens here too -- this is
    # to reduce memory usage (storing hundreds of images' worth of features in memory is a surefire
    # way to fry your beloved computer).
    #
    # TODO: find some sensible way to remove this duplicated code
    feat_chain = []
    package = "features"
    for stage in config[package]:
        # import module and fetch the specified method
        module = ".".join((package, stage["name"]))
        mod = importlib.import_module(module)
        method = getattr(mod, stage["method"])

        # fetch the parameters, generate all parameter possibilities and chuck them into a lambda
        # function so that you only need to call: "method(image)".
        if "parameters" in stage:
            for args in itertools.product(*stage["parameters"]):
                feat_chain.append({
                    "description": stage["description"],
                    "method": lambda im: method(im, *args),
                    "parameters": args
                })
        else:
            feat_chain.append({
                "description": stage["description"],
                "method": method
            })

    for i in range(0, len(ground_truths)):
        # apply preprocessing
        gt = ground_truths[i]["img"]
        im = images[i]["img"]
        for link in prep_chain:
            gt = link["method"](gt)
            im = link["method"](im)

        for feature in feat_chain:
            # import modules and fetch the specified methods
            for stage in config[package]:
                module = ".".join((package, stage["name"]))
                mod = importlib.import_module(module)
                method = getattr(mod, stage["method"])

    # TODO: apply functions to image and analyse results
    # TODO: write way of comparing computed features and ground truth
    # TODO: logging or command-line output
    # TODO: DRY
    # TODO: replace current file list with config file or something
    # TODO: improve docs and commenting
    # TODO: variable naming

    #for i in range(0, len(features["ground_truth"])):
    #    feature_type = features["ground_truth"][i]["type"]
    #    print(feature_type)

    #    for j in range(0, len(features["ground_truth"][i]["features"])):
    #        ground_truth = features["ground_truth"][i]["features"][j]
    #        image = features["images"][i]["features"][j]

    #        mse = ((ground_truth["image_features"] - image["image_features"]) ** 2).mean(axis=None)
    #        print("frequency: %s, angle: %s = error: %f" % (image["parameters"][0], image["parameters"][1], mse))
