#! /usr/bin/env python2

import copy
import glob
import importlib
import itertools
import multiprocessing
import os
import sys

import dill
import json
import yaml

from skimage.io import imread


def _imshow(*images):
    from matplotlib import pyplot as plt

    fig = plt.figure()
    for i in range(0, len(images)):
        im = images[i]
        if im.shape[0] > im.shape[1]:
            ax = fig.add_subplot(1, len(images), i + 1)
        else:
            ax = fig.add_subplot(len(images), 1, i + 1)

        cax = ax.imshow(im, cmap=plt.cm.cubehelix)
        fig.colorbar(cax)

    plt.show()


def read_config(path):
    """
    Read a YAML config file into a data structure and return it. Does what it says on the tin.
    """
    with open(path) as conf_in:
        config = yaml.load(conf_in)
    return config


def gen_file_list(ls):
    file_list = []
    for _glob in ls:
        f_list = glob.glob(os.path.abspath(_glob))
        if len(f_list) > 0:
            for path in f_list:
                file_list.extend(f_list)
        else:
            file_list.append(_glob)
    return file_list


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


def extract_features_and_compare(feature, clustering, shadow_seg, ground_truth, image, tests):
    feat_im = feature["method"](image)
    clus_im = clustering["method"](feat_im)
    shadow = shadow_seg["method"](image, clus_im)

    result = {
        "features": {
            "name": feature["name"],
            "description": feature["description"]
        },
        "clustering": {
            "name": clustering["name"],
            "description": clustering["description"]
        },
        "shadow_seg": {
            "name": shadow_seg["name"],
            "description": shadow_seg["description"]
        },
        "results": {}
    }

    if "parameters" in feature:
        result["features"].update({"parameters": feature["parameters"]})
    if "parameters" in clustering:
        result["clustering"].update({"parameters": clustering["parameters"]})
    if "parameters" in shadow_seg:
        result["shadow_seg"].update({"parameters": shadow_seg["parameters"]})

    for test in tests:
        test_method = test["method"]
        name = ".".join([test["name"], test["method"].__name__])
        r_upd = {name: test_method(ground_truth, shadow)}
        result["results"].update(r_upd)

    return result


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
    if len(sys.argv) < 3:
        CONFIG_FILES = os.path.join(os.getcwd(), "config/config.yml")
        FILE_LIST = os.path.join(os.getcwd(), "config/files.yml")
    else:
        CONFIG_FILES = os.path.abspath(sys.argv[1])
        FILE_LIST = os.path.abspath(sys.argv[2])

    # Append current directory to system path
    cur_dir = os.path.dirname(__file__)
    sys.path.append(cur_dir)

    # Get configuration
    config = read_config(CONFIG_FILES)
    file_list = read_config(FILE_LIST)

    # Get input file list and make sure everything's hunky-dory
    ground_truths = gen_file_list(file_list["input"]["ground_truths"])
    images = gen_file_list(file_list["input"]["images"])
    assert len(ground_truths) == len(images)

    # Create the function chains for image preprocessing and feature extraction
    prep_chain = construct_function_list(config, "preprocess")
    feat_chain = construct_function_list(config, "features")
    cluster_chain = construct_function_list(config, "cluster")
    shadow_chain = construct_function_list(config, "shadow")
    testing_chain = construct_function_list(config, "metrics")

    # Here the function chains in prep_chain and feat_chain are applied. Each function in the
    # preprocessing chain is applied to each image in series, and the preprocessed images are then
    # run through each feature extraction algorithm individually. These results are then tested.
    results = {}
    pl = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    package = "tests"
    for i in range(0, len(ground_truths)):
        # create result array
        results[images[i]] = []

        # read images
        gt = imread(ground_truths[i], as_grey=True)
        im = imread(images[i])

        # apply preprocessing chain to images
        for link in prep_chain:
            im = link["method"](im)
        assert len(gt) == len(im)

        # extract features and compare ground truth
        for feature in feat_chain:
            for clustering in cluster_chain:
                for shadow_seg in shadow_chain:
                    if "parameters" in feature:
                        sys.stderr.write("Dispatching: %s%s\n" % (feature["name"], feature["parameters"]))
                    else:
                        sys.stderr.write("Dispatching: %s()\n" % (feature["name"]))

                    results[images[i]].append(
                        apply_async(
                            pl,
                            extract_features_and_compare,
                            (feature, clustering, shadow_seg, gt, im, testing_chain)
                        )
                    )

    # Close the pool, and wait for all of the subprocesses to finish whatever they were doing
    sys.stderr.write("Waiting for subprocesses to finish...\n")
    pl.close()
    pl.join()

    # Get results now that processing has finished
    n_combinations = len(results[results.keys()[0]])
    for i in range(0, n_combinations):
        for k, v in results.items():
            results[k] = [r.get() for r in v]

    #results = sorted([r.get() for r in results], key=lambda x: [x[s_key] for s_key in sort_keys])

    # Combine and sort results
    #sort_keys = config["sort"]["keys"]

    # Pretty-print results
    print(json.dumps(results, indent=4))
