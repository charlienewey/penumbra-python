from sklearn.cluster import MiniBatchKMeans


def cluster_to_labels(image_features, n_clusters=15, batch_size=100):
    # get texture features
    original_shape = image_features.shape
    feats_r = image_features.reshape(-1, 1)

    # cluster the texture features, reusing initialised centres if already calculated
    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
    clus = km.fit(feats_r)

    # calculate labels
    labels = clus.labels_

    # TODO: calculate cluster centroids and use as initialisation for next frame?

    # reshape label arrays
    labels = labels.reshape(original_shape[0], original_shape[1])

    return labels
