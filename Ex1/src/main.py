import random
import matplotlib.pyplot as plt

import numpy as np
from Ex1.src.Bandit import Bandit
from Ex1.src.Action import Action

def q1(number_of_arms = 10, iterations = 100):
    true_values = np.random.normal(0, 1, number_of_arms)
    print(true_values)
    B = Bandit()
    for i in true_values:
        B.add_action(Action(i, 0))
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

if __name__=="__main__":
    q1(10,1000)