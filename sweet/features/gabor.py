from skimage.filters import gabor_filter


def squared_error(image_1, image_2):
    pass


def correlation(ground_truth, features):
    pass


def gabor_filter(image, spatial_frequencies, angles):
    features = {
        "type": __name__.split(".")[-1],
        "features": []
    }

    for freq in spatial_frequencies:
        for theta in angles:
            print("frequency: %1.1f, angle: %d" % (freq, theta))

            features["features"].append({
                "parameters": [freq, theta],
                # the "gabor_filter" function has two components, 0 (real), and 1 (imaginary)
                # TODO: square the imaginary part and add to the real part
                "image_features": gabor_filter(image, freq, theta)[0]
            })

    return features
