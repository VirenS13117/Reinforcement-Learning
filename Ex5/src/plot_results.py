import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Tuple, Dict, Set, Callable
from matplotlib import cm
import math


def plot_performance(trial_data_list, clrs, algo_titles):
    fig, plt1 = plt.subplots(1, figsize=(8, 8), facecolor="white")
    def subplot_performance(clrs: List[str], ydata: List[int], trial_data_list, s_plt):
        for i in range(len(clrs)):
            xdata = np.average(trial_data_list[i], axis=0)
            s_plt.plot(xdata, ydata, color=clrs[i], label=algo_titles[i])
            xstderr = np.std(trial_data_list[i], axis=0)
            xstderr *= 1 / math.sqrt(trial_data_list[i].shape[1])
            xstderr *= 1.96
            s_plt.fill_betweenx(ydata, np.subtract(xdata, xstderr), np.add(xdata, xstderr),  alpha=0.2, color=clrs[i])
        return

    ydata = [x for x in range(1, trial_data_list.shape[2] + 1)]
    subplot_performance(clrs, ydata, trial_data_list, plt1)
    # plt1.set_xlim(0, 10000)
    plt1.legend(loc="upper right")
    plt1.set_xlabel("Steps")
    plt1.set_ylabel("Episodes")
    plt1.set_title("Performance plot:" + " Episodes = " + str(trial_data_list.shape[2]) + "," + " Trials = " + str(trial_data_list.shape[1]))
    plt.show()

    return

def plot_histogram(histogram_data, color, title):
    x = histogram_data
    plt.hist(x, density=True, bins=30)  # `density=False` would make counts
    # plt.xlim(-40, 10)
    plt.ylabel('count')
    plt.xlabel(title)
    plt.show()
    return