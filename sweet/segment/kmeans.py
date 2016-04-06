def cluster_labels(image_features, n_clusters=15):
    # blur and take local maxima
    #image = gaussian_filter(image, sigma=8)
    #blur_image = ndi.maximum_filter(image, size=3)

    # get texture features
    #num_feats, feats = lbp(blur_image, n=5, method="uniform")
    original_shape = image_features.shape
    feats_r = image_features.reshape(-1, num_feats)

    # cluster the texture features, reusing initialised centres if already calculat
    km = k_means(n_clusters=n_clusters)
    clus = km.fit(feats_r)

    # copy relevant attributes
    labels = clus.labels_

    # reshape label arrays
    labels = labels.reshape(original_shape[0], original_shape[1])

    return labels
