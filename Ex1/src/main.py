import random
import matplotlib.pyplot as plt

import numpy as np
from Ex1.src.Bandit import Bandit
from Ex1.src.Action import Action


def q1(number_of_arms=10, iterations=100):
    true_values = np.random.normal(0, 1, number_of_arms)
    print(true_values)
    B = Bandit(0)
    for i in true_values:
        B.add_action(Action(i, 0, 1))
    arm_rewards = [[] for i in range(number_of_arms)]
    for i, action in enumerate(B.actions):
        for itr in range(iterations):
            arm_rewards[i].append(action.play_action())
    print(arm_rewards)
    fig = plt.figure()
    # Create an axes instance
    ax = fig.gca()
    plt.ylabel('Reward Distribution')
    plt.xlabel('Action')
    # Create the violinplot
    violinplot = ax.violinplot(arm_rewards)
    plt.show()
    return


def q2(eps, number_of_arms=10, runs=2000, steps=10000):
    xdata = [x for x in range(steps)]
    for epsilon in eps:
        arm_rewards = [[] for i in range(runs)]
        for i in range(runs):
            true_values = np.random.normal(0, 1, number_of_arms)
            B = Bandit(epsilon)
            for j in true_values:
                B.add_action(Action(j, 0, 1))
            for s in range(steps):
                best_action = B.choose_action()
                reward = best_action.play_action()
                best_action.sample_average_update(reward)
                arm_rewards[i].append(reward)

        ydata = np.average(arm_rewards, axis=0)
        plt.plot(xdata, ydata, linewidth=2, label=str(epsilon))
    plt.ylabel('Average reward')
    plt.xlabel('Steps')
    plt.title("Average reward per step with different epsilon values")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # q1(10,10000)
    q2([0, 0.1, 0.01])
    # q2(0.01)
