import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import numpy as np
from Ex1.src.Bandit import Bandit
from Ex1.src.Action import Action


def q4(number_of_arms=10, iterations=100):
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


def run_bandit_trial(B, eps, steps, optimal_curve=False, optimal_action_trial=False):
    trial_rewards = []
    for i in range(steps):
        if optimal_action_trial:
            best_action = B.optimal_action
        else:
            best_action = B.choose_action(eps)
        reward = best_action.play_action()
        best_action.sample_average_update(reward)
        if optimal_curve:
            trial_rewards.append(B.optimal_action == best_action)
        else:
            trial_rewards.append(reward)
    return trial_rewards


def plot_curve(runs, eps, steps, average_blocks, total_optimal_rewards, optimal_curve=False):
    xdata = [x for x in range(steps)]
    for i in range(len(eps)):
        ydata = np.average(average_blocks[i], axis=0)
        ydata_std = np.std(average_blocks[i], axis=0)
        ydata_std /= np.sqrt(runs)
        ydata_std *= 1.96
        plt.plot(xdata, ydata, linewidth=2, label=r'$\epsilon$ = ' + str(eps[i]))
        plt.fill_between(xdata, np.subtract(ydata, ydata_std), np.add(ydata, ydata_std), alpha=0.2)

    ydata = np.average(total_optimal_rewards, axis=0)

    plt.plot(xdata, ydata, linewidth=1)
    if optimal_curve:
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

    if optimal_curve:
        plt.ylabel('% Optimal Action')
        plt.title("Optimal action % per step with different epsilon values")
    else:
        plt.ylabel('Average reward')
        plt.title("Average performance of $\epsilon$ greedy method on 10 arm testbed")
    plt.xlabel('Steps')

    plt.legend()
    plt.show()


def q6(eps, optimal_curve=False, number_of_arms=10, runs=200, steps=1000):
    total_rewards = []
    total_optimal_rewards = []
    for i in range(runs):
        true_values = np.random.normal(0, 1, number_of_arms)
        max_value = max(true_values)
        B = Bandit()
        rewards_per_trial = []
        for j in true_values:
            if j == max_value:
                B.add_action(Action(j, 0, 1), True)
            else:
                B.add_action(Action(j, 0, 1), False)
        optimal_rewards = run_bandit_trial(B, 0, steps, optimal_curve, optimal_action_trial=True)
        total_optimal_rewards.append(optimal_rewards)
        for epsilon in range(len(eps)):
            B.reset()
            rewards_per_eps = run_bandit_trial(B, eps[epsilon], steps, optimal_curve, optimal_action_trial=False)
            rewards_per_trial.append(rewards_per_eps)
        total_rewards.append(rewards_per_trial)

    average_blocks = []
    for i in range(len(eps)):
        eps_block = []
        for j in range(runs):
            eps_block.append(total_rewards[j][i])
        average_blocks.append(eps_block)

    plot_curve(runs, eps, steps, average_blocks, total_optimal_rewards, optimal_curve)


if __name__ == "__main__":
    epsilon_list = [0, 0.1, 0.01]
    number_of_arms = 10
    runs = 2000
    step = 1000
    optimal_curve = True
    # q4(10,10000)
    q6(epsilon_list, optimal_curve, number_of_arms, runs, step)
    # q2(0.01)
