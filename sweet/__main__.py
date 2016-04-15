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
            file_list.extend(f_list)
        else:
            file_list.append(_glob)
    return sorted(file_list)


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

    output = {}
    for test in tests:
        test_method = test["method"]
        name = ".".join([test["name"], test["method"].__name__])
        output[name] = test_method(ground_truth, shadow)

    return output


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

    total_processes = (len(feat_chain) * len(cluster_chain) * len(shadow_chain)) * len(images)
    print("Dispatching %d processes..." % (total_processes))

    # Here the function chains in prep_chain and feat_chain are applied. Each function in the
    # preprocessing chain is applied to each image in series, and the preprocessed images are then
    # run through each feature extraction algorithm individually. These results are then tested.
    output = []
    pl = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    # extract features and compare ground truth
    for feature in feat_chain:
        for clustering in cluster_chain:
            for shadow_seg in shadow_chain:
                # set up data structure for results
                combination = {
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
                    "results": []
                }

                if "parameters" in feature:
                    combination["features"].update({"parameters": feature["parameters"]})
                if "parameters" in clustering:
                    combination["clustering"].update({"parameters": clustering["parameters"]})
                if "parameters" in shadow_seg:
                    combination["shadow_seg"].update({"parameters": shadow_seg["parameters"]})

                output.append(combination)

                # iterate through files, testing each pair of ground truth/input
                for i in range(0, len(ground_truths)):
                    # read image
                    gt = imread(ground_truths[i], as_grey=True)
                    im = imread(images[i])

                    # apply preprocessing chain to image
                    for link in prep_chain:
                        im = link["method"](im)
                    assert len(gt) == len(im)

                    r = apply_async(
                        pl,
                        extract_features_and_compare,
                        (feature, clustering, shadow_seg, gt, im, testing_chain)
                    )

                    combination["results"].append(r)

    # Close the pool, and wait for all of the subprocesses to finish whatever they were doing
    sys.stderr.write("Dispatched all tasks, waiting for processes to finish...\n")
    pl.close()
    pl.join()

    # Get results now that processing has finished
    for combination in output:
        output_list = combination["results"]
        fetched_output_list = []
        for result in output_list:
            fetched_output_list.append(result.get())
        combination["results"] = fetched_output_list

    # Reduce the list of dictionaries into a single dictionary
    for combination in output:
        results = {}
        for result in combination["results"]:
            for test, value in result.items():
                if test in results:
                    results[test] += value
                else:
                    results[test] = value

        # this section is a bit hacky, just calculates rates from the results
        tp = float(results["binary_classification.true_positives"])
        tn = float(results["binary_classification.true_negatives"])
        fp = float(results["binary_classification.false_positives"])
        fn = float(results["binary_classification.false_negatives"])

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tnr = tn / (tn + fp)
        fnr = fn / (fn + tp)

        results = {}

        results["tpr"] = tpr
        results["fpr"] = fpr
        results["tnr"] = tnr
        results["fnr"] = fnr

        combination["results"] = results

    output = sorted(output, key=lambda x: x["results"]["tpr"] + x["results"]["tnr"])

    # Pretty-print results
    print(json.dumps(output, indent=4))
