# -*- coding: utf-8 -*-
"""
Util functions to visualize plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_distortions(k_upper: int, distortions: list):
    """
    visualize distortions in a scatter plot
    args:
        - k_upper: the upper bound of k, inclusive
        - distortions: a list of sse/distortions resulting from all k's
    """
    k_lst = range(1, k_upper+1)
    
    plt.figure(figsize=(6,4))
    plt.plot(k_lst, distortions, color='tomato', marker='x')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.xticks(k_lst)
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    return

def plot_label_distr(labels: np.ndarray):
    """
    visualize distribution of labels in a bar chart
    args:
        - labels: labels of each data point, resulting from kmeans
    """
    # find number of unqiue labels and their counts
    unique_labels, label_count = np.unique(labels, return_counts=True)
    plt.figure(figsize=(6,4))
    plt.bar(unique_labels, label_count, color='tomato')
    plt.xticks(ticks=unique_labels, labels=unique_labels)
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.title('Distribution of Cluster Counts')
    plt.show
    return 
    
def plot_return(
        labels: np.ndarray,
        returns: np.ndarray,
        k: int,
        nrows: int,
        ncols: int
        ):
    """
    visualize distibution of return in a histogram
    args:
        - labels: cluster labels of each data point, resulting from kmeans
        - returns: annualized next day returns for each data point
        - k: number of clusters == len(np.unique(labels))
        - nrows: number of subplots in a row
        - ncols: number of subplots in a col
    """
    fig, axs = plt.subplots(nrows, ncols, figsize=(10,10))
    for cluster_num in range(k):
        row = cluster_num // nrows
        col = cluster_num % ncols
        # find indices s.t. label == current cluster_num
        cluster_ind = np.argwhere(labels==cluster_num)
        # add histogram
        axs[row,col].hist(returns[cluster_ind], density=True,
                          color='tomato', edgecolor='black')
        # set title
        axs[row, col].set_title('Cluster ' + str(cluster_num))
    # set x and y labels
    for ax in axs.flat:
        ax.set(xlabel='return', ylabel='density')
    plt.show()
    return
    