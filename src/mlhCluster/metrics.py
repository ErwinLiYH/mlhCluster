from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import scipy.sparse as sp

def distortion_score(X, labels, metric="cosine"):
    # Encode labels to get unique centers and groups
    le = LabelEncoder()
    le.fit(labels)
    unique_labels = le.classes_

    # Sum of the distortions
    distortion = 0

    # Loop through each label (center) to compute the centroid
    for current_label in unique_labels:
        # Mask the instances that belong to the current label
        mask = labels == current_label
        instances = X[mask]

        # Compute the center of these instances
        center = instances.mean(axis=0)

        # NOTE: csc_matrix and csr_matrix mean returns a 2D array, numpy.mean
        # returns an array of 1 dimension less than the input. We expect
        # instances to be a 2D array, therefore to do pairwise computation we
        # require center to be a 2D array with a single row (the center).
        # See #370 for more detail.
        if not sp.issparse(instances):
            center = np.array([center])

        # Compute the square distances from the instances to the center
        distances = pairwise_distances(instances, center, metric=metric)
        distances = distances ** 2

        # Add the sum of square distance to the distortion
        distortion += distances.sum()

    return distortion
