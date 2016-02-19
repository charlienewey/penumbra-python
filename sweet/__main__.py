#! /usr/bin/env python2

import copy
import importlib
import itertools
import multiprocessing
import os
import sys

import cv2
import dill
import json
import yaml


def imshow(*images):
    """
    Display an image in a window. Mostly used for debugging at the moment.
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
    """
    Read a YAML config file into a data structure and return it. Does what it says on the tin.
    """
    with open(CONFIG_FILE) as conf_in:
        config = yaml.load(conf_in)
    return config


def make_method(method, *args):
    """
    Create a higher-order partial function calling 'method' with the parameters '*args'. This has
    the effect of creating a function that takes one parameter -- an image. This is used as part of
    the processing pipeline.

    This is used instead of lambda im: method(im, *args), because lambda functions are lazily
    evaluated in Python, and that causes all sorts of headaches.
    """
    def new_method(image):
        return method(image, *args)
    return new_method


def run_dill_encoded(what):
    """
    Run a function that's been encoded with Dill (this allows me to do multiprocessing).

    http://stackoverflow.com/a/24673524
    """
    fun, args = dill.loads(what)
    return fun(*args)


def apply_async(pool, fun, args):
    """
    Apply a function to a pool asynchronously -- wrapping around the run_dill_encoded(...) function.

    http://stackoverflow.com/a/24673524
    """
    return pool.apply_async(run_dill_encoded, (dill.dumps((fun, args)),))


def extract_features_and_compare(package, feature, ground_truth, image):
    feat_gt = feature["method"](ground_truth)
    feat_im = feature["method"](image)

    results = []
    for test in package:
        test_method = getattr(feature["module"], test["name"])
        result = {
            "name": feature["name"],
            "description": feature["description"],
            test["name"]: test_method(feat_gt, feat_im),
            "parameters": feature["parameters"]
        }
        results.append(result)
    return results


def construct_function_list(config, package):
    # This is probably a bit indecipherable if you aren't familiar with Python introspection, so...
    # sorry for that. This section imports all of the specified modules and their functions, fetches
    # the functions, and appends them to a list along with their parameters -- effectively creating
    # a processing pipeline. This pipeline can then be applied multiple times to different image
    # pairs, making processing more efficient.
    #
    # This way of optimising the pipeline keeps memory usage as low as possible, and also lends
    # itself well to parallelisation.
    #
    # * https://docs.python.org/2/library/importlib.html
    # * https://docs.python.org/2/library/functions.html#getattr
    # * https://docs.python.org/2.7/reference/compound_stmts.html?highlight=*identifier#function-definitions
    # * http://stackoverflow.com/a/400823
    chain = []
    for link in config[package]:
        # import module and fetch the specified method
        module_name = ".".join((package, link["name"]))
        module = importlib.import_module(module_name)
        method = getattr(module, link["method"])

        # fetch the parameters, generate all parameter possibilities and chuck them into a lambda
        # function so that you only need to call: "method(image)".
        if "parameters" in link:
            for parameters in itertools.product(*link["parameters"]):
                chain.append({
                    "name": link["name"],
                    "description": link["description"],
                    "method": make_method(method, *parameters),
                    "module": module,
                    "parameters": parameters
                })
        else:
            chain.append({
                "name": link["name"],
                "description": link["description"],
                "method": method,
                "module": module
            })

    return chain


if __name__ == "__main__":
    # Set constants here - will later replace with command-line opts
    CONFIG_FILE = os.path.join(os.getcwd(), "config/config.yml")

    # Append current directory to system path
    cur_dir = os.path.dirname(__file__)
    sys.path.append(cur_dir)

    # Get configuration
    config = read_config(CONFIG_FILE)

    # Read input files and make sure everything's hunky-dory
    # TODO: replace this with something a bit more sensible (list of files, etc)
    ground_truths = [cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)]
    images = [cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)]
    assert len(ground_truths) == len(images)

    # Create the function chains for image preprocessing and feature extraction
    prep_chain = construct_function_list(config, "preprocess")
    feat_chain = construct_function_list(config, "features")

    # Here the function chains in prep_chain and feat_chain are applied. Each function in the
    # preprocessing chain is applied to each image in series, and the preprocessed images are then
    # run through each feature extraction algorithm individually. These results are then tested.
    results = []
    pl = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    package = "tests"
    for i in range(0, len(ground_truths)):
        # apply preprocessing chain to images
        gt = ground_truths[i]
        im = images[i]
        for link in prep_chain:
            gt = link["method"](gt)
            im = link["method"](im)
        assert len(gt) == len(im)

        # extract features and compare ground truth
        for j in range(0, len(gt)):
            gtj = gt[j]
            imj = im[j]
            for feature in feat_chain:
                sys.stderr.write("Dispatching: %s%s\n" % (feature["name"], feature["parameters"]))
                results.append(apply_async(
                        pl,
                        extract_features_and_compare,
                        (config[package], feature, gtj, imj))
                )

    # Close the pool, and wait for all of the subprocesses to finish whatever they were doing
    sys.stderr.write("Waiting for subprocesses to finish...\n")
    pl.close()
    pl.join()

    # Get results now that processing has finished
    results = list(itertools.chain(*[r.get() for r in results]))

    # Pretty-print results
    print(json.dumps(results, indent=4))

    # TODO: replace current file list with config file or something
