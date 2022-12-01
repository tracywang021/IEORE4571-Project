# -*- coding: utf-8 -*-
"""
Util functions to visualize plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_distortions(
        k_upper: int,
        distortions: list,
        fig_size: tuple,
        file_name: str=None
        ):
    """
    visualize distortions in a scatter plot
    args:
        - k_upper: the upper bound of k, inclusive
        - distortions: a list of sse/distortions resulting from all k's
        - fig_size: the size of the plot
        - file_name: save plot as file_name 'xxx.png'
    """
    k_lst = range(1, k_upper+1)
    
    plt.figure(figsize=fig_size)
    plt.plot(k_lst, distortions, color='tomato', marker='x')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.xticks(k_lst)
    plt.title('The Elbow Method showing the optimal k')
    # save plot if a file_name is passed in
    if file_name:
        parent_path = 'imgs/'
        save_path = parent_path + file_name
        plt.savefig(save_path)
    else:
        plt.show()
    return

def plot_label_distr(labels: np.ndarray, fig_size: tuple, file_name: str=None):
    """
    visualize distribution of labels in a bar chart
    args:
        - labels: labels of each data point, resulting from kmeans
        - fig_size: the size of the plot
        - file_name: save plot as file_name 'xxx.png'
    """
    # find number of unqiue labels and their counts
    unique_labels, label_count = np.unique(labels, return_counts=True)
    plt.figure(figsize=fig_size)
    plt.bar(unique_labels, label_count, color='tomato')
    plt.xticks(ticks=unique_labels, labels=unique_labels)
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.title('Distribution of Cluster Counts')
    # save plot if a file_name is passed in
    if file_name:
        parent_path = 'imgs/'
        save_path = parent_path + file_name
        plt.savefig(save_path)
    else:
        plt.show()
    return 
    
def plot_return(
        labels: np.ndarray,
        returns: np.ndarray,
        k: int,
        nrows: int,
        ncols: int,
        fig_size: tuple,
        file_name: str=None
        ):
    """
    visualize distibution of return in a histogram
    args:
        - labels: cluster labels of each data point, resulting from kmeans
        - returns: annualized next day returns for each data point
        - k: number of clusters == len(np.unique(labels))
        - nrows: number of subplots in a row
        - ncols: number of subplots in a col
        - fig_size: the size of the plot
        - file_name: save plot as file_name 'xxx.png'
    """
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                            figsize=fig_size)
    for cluster_num in range(k):
        col = cluster_num % ncols
        row = (cluster_num-col)//ncols
        # find indices s.t. label == current cluster_num
        cluster_ind = np.argwhere(labels==cluster_num)
        # add histogram
        axs[row,col].hist(returns[cluster_ind], density=True,
                          color='tomato', edgecolor='black')
        # set title
        axs[row, col].set_title('Cluster ' + str(cluster_num))
        # set y labels
        if col % ncols == 0:
            axs[row, col].set_ylabel('density')
        if (row+1) % nrows == 0:
            axs[row, col].set_xlabel('return')
        # show ticklabels
        axs[row, col].set_visible(True)
        # save plot if a file_name is passed in
    if file_name:
        parent_path = 'imgs/'
        save_path = parent_path + file_name
        plt.savefig(save_path)
    else:
        plt.show()
    return
    