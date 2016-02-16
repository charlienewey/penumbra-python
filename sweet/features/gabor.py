

def squared_error(image_1, image_2):
    pass


def correlation(ground_truth, features):
    pass


def features(image, spatial_frequencies, angles):
    # TODO: finish
    features = []
    for freq in spatial_frequencies:
        for theta in angles:
            features.append(gabor_filter(image, freq, angle))
