# -*- coding: utf-8 -*-
"""
Util functions to visualize plots
"""

import matplotlib.pyplot as plt

def plot_distortions(k_upper: int, distortions: list):
    """
    visualize distortions in a scatter plot
    args:
        - k_upper: the upper bound of k, inclusive
        - distortions: a list of sse/distortions resulting from all k's
    """
    k_lst = range(1, k_upper+1)
    
    plt.figure(figsize=(8,6))
    plt.plot(k_lst, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.xticks(k_lst)
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    return
    