# -*- coding: utf-8 -*-
"""
Util functions for modelling
"""

import numpy as np
from sklearn.cluster import KMeans


def group_kmeans(k_upper: int, X: np.ndarray) -> list:
    """
    perform a group of kmeans clustering with different k's'
    args:
        - k_upper: the upper bound of k, inclusive
        - X: attributes to feed in the algorithm
    return:
        - sse: a list of sse/distortions resulting from all k's
    """
    sse = []
    # a list of k's that will be used clustering
    k_lst = range(1, k_upper+1)
    for k in k_lst:
        model = KMeans(n_clusters=k, random_state=0)
        model.fit(X)
        sse.append(model.inertia_)
    return sse

def single_kmeans(k: int, X: np.ndarray) -> np.ndarray:
    """
    perform kmeans clustering with given k
    args:
        - k: number of clusters
        - X: attributes to feed in the algorithm
    return:
        - labels of each data point, resulting from kmeans
    """
    model = KMeans(n_clusters=k, random_state=0)
    model.fit(X)
    return model.labels_
