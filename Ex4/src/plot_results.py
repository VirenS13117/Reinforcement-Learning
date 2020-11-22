import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Tuple, Dict, Set, Callable
from matplotlib import cm
from Ex4.src.Policy import Policy
from Ex4.src.randomPolicy import RandomPolicy
import math


def plot_value_fn(vs: Dict, title: str, elevation: int = 150):
    def plot_image(plt: plt, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, title: str, r: int = 1, c: int = 1,
                   plot_index: int = 1):
        ax = plt.subplot(r, c, plot_index, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, elevation)
        return

    plt.figure(figsize=(18, 6), facecolor='white')

    # Make x, y axis
    X, Y = np.meshgrid(np.arange(12, 21 + 1), np.arange(1, 10 + 1))

    # Evaluate Z for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: vs[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: vs[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    # plot images
    plot_image(plt, X, Y, Z_noace, "{} (No Usable Ace)".format(title), 1, 2, 1)
    plot_image(plt, X, Y, Z_ace, "{} (Usable Ace)".format(title), 1, 2, 2)

    # plot display
    plt.show()


def plot_policy(qpi: Dict, pi: Dict, title: str):
    def plot_image(nrows: int, ncols: int, plt_index: int, plt: plt, xlabel: str, ylabel: str, title: str,
                   heatmap: np.ndarray):
        ax = plt.subplot(nrows, ncols, plt_index)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.imshow(heatmap)
        return
    plt.figure(figsize=(10, 10), facecolor='white')
    Z_noace = np.zeros((11, 10))
    Z_ace = np.zeros((11, 10))
    for i, x in enumerate(range(11, 21 + 1)):
        for j, y in enumerate(range(1, 10 + 1)):
            Z_noace[i][j] = qpi.get((x, y, False), pi.get_action((x, y, False)))
            Z_ace[i][j] = qpi.get((x, y, True), pi.get_action((x, y, True)))
    plot_image(1, 2, 1, plt, "", "", "{} (No Usable Ace)".format(title), np.flipud(Z_noace))
    plot_image(1, 2, 2, plt, "", "", "{} (Usable Ace)".format(title), np.flipud(Z_ace))
    plt.show()

def plot_policy_values(env, pi: Dict, title: str, Q: Dict = None):
    offset = 0.3
    fig, ax = plt.subplots(figsize=(6,6), facecolor="white")
    x_pos = []
    y_pos = []
    x_dir = []
    y_dir = []
    start = env.start
    end = env.end
    rows = end[0]+1-start[0]
    cols = end[1]+1-start[1]
    for state in env.states:
        if state != env.target:
            y_pos.append(state[0] + offset)
            x_pos.append(state[1] + offset)
            move = env.direction[env.get_action_name(pi[state])]
            x_dir.append(move[1])
            y_dir.append(move[0])
            if Q:
                ax.annotate(str(np.round(Q[state][pi[state]],2)), (state[1], state[0]))
    for obstacle in env.blocks:
        y_pos.append(obstacle[0] + offset)
        x_pos.append(obstacle[1] + offset)
        x_dir.append(0)
        y_dir.append(0)
        if Q:
            ax.annotate(str(np.round(Q[obstacle][pi[state]],2)), (obstacle[1], obstacle[0]))

    y_pos.append(env.target[0] + offset)
    x_pos.append(env.target[1] + offset)
    x_dir.append(0)
    y_dir.append(0)
    if Q:
        ax.annotate(str(np.round(Q[env.target][pi[state]],2)), (env.target[1], env.target[0]))
    ax.quiver(x_pos, y_pos, x_dir, y_dir)
    xticks = [i for i in range(cols)]
    ax.set_xlim(0, cols)
    ax.set_xticks(xticks)
    yticks = [i for i in range(rows)]
    ax.set_ylim(0, rows)
    ax.set_yticks(yticks[::-1])
    ax.grid()
    ax.set_title(title)
    plt.show()
    return

def plot_performance(eps: List[float], clrs: List[str], G: np.ndarray):
    fig, plt1 = plt.subplots(1, figsize=(8, 8), facecolor="white")

    def subplot_performance(eps: List[float], clrs: List[str], xdata: List[int], G, s_plt):
        for i, param in enumerate(eps):
            ydata = np.average(G[i], axis=0)
            s_plt.plot(xdata, ydata, color=clrs[i], label="$\epsilon$ = " + str(param))
            ystderr = np.std(G[i], axis=0)
            ystderr *= 1 / math.sqrt(G.shape[1])
            ystderr *= 1.96
            s_plt.fill_between(xdata, np.subtract(ydata, ystderr), np.add(ydata, ystderr), alpha=0.2, color=clrs[i])
        G_Max = np.amax(G)
        ydata = [G_Max for x in range(G.shape[2])]
        s_plt.plot(xdata, ydata, color="black", label="Upper Bound")
        return
    xdata = [x for x in range(G.shape[2])]
    subplot_performance(eps, clrs, xdata, G, plt1)
    plt1.legend(loc="lower right")
    plt1.set_xlabel("Episodes")
    plt1.set_ylabel("Average Return")
    plt1.set_title("Performance plot:" + " Episodes = " + str(G.shape[2]) + "," + " Trials = " + str(G.shape[1]))
    plt.show()

    return
