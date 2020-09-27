import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import numpy as np
from Ex1.src.Bandit import Bandit
from Ex1.src.Action import Action
from Ex1.src.Params import Params


def q4(number_of_arms=10, iterations=100):
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
    ax.yaxis.grid(True)
    plt.ylabel('Reward Distribution')
    plt.xlabel('Action')
    # Create the violinplot
    violinplot = ax.violinplot(arm_rewards,showmeans = True)
    plt.show()
    return


def run_bandit_trial(B, eps, ucb, c, steps, stationary=True, optimal_curve_plot=False, sample_average=True, alpha=1):
    trial_rewards = []
    optimal_rewards = []
    for i in range(steps):
        best_action = B.choose_action(eps, ucb, c, i+1)
        reward = best_action.play_action()
        if sample_average:
            best_action.exponential_recency_weighted_average(reward, 1/(best_action.occurrence+1))
        else:
            best_action.exponential_recency_weighted_average(reward, alpha)
        if not stationary:
            B.update()
        if optimal_curve_plot:
            trial_rewards.append(B.optimal_action == best_action)
            optimal_rewards.append(1)
        else:
            trial_rewards.append(reward)
            optimal_rewards.append(B.optimal_action.true_value)
    return trial_rewards, optimal_rewards


def plot_curve(runs, param_list, steps, average_blocks, total_optimal_rewards, optimal_curve=False):
    xdata = [x for x in range(steps)]
    for i in range(len(param_list)):
        ydata = np.average(average_blocks[i], axis=0)
        ydata_std = np.std(average_blocks[i], axis=0)
        ydata_std /= np.sqrt(runs)
        ydata_std *= 1.96
        label_set = ''
        if param_list[i].sample_average:
            label_set += r'$\epsilon$ = ' + str(param_list[i].epsilon)
            # label_set += r'; $Q_{1}$ = ' + str(param_list[i].initial_values[0])
            # label_set += '; sample average'
        else:
            label_set += r'$\epsilon$ = ' + str(param_list[i].epsilon)
            label_set += r'; $Q_{1}$ = ' + str(param_list[i].initial_values[0])
            label_set += '; constant step size with '+r'$\alpha$ = ' + str(param_list[i].alpha)
        # if param_list[i].ucb:
        #     label_set = "UCB c = "+str(param_list[i].confidence_level)
        # else:
        #     label_set = "$\epsilon$ greedy $\epsilon$ = "+str(param_list[i].epsilon)
        plt.plot(xdata, ydata, linewidth=2, label=label_set)
        plt.fill_between(xdata, np.subtract(ydata, ydata_std), np.add(ydata, ydata_std), alpha=0.2)
    ydata = np.average(total_optimal_rewards[i], axis=0)
    plt.plot(xdata, ydata, linewidth=1)

    if optimal_curve:
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

    if optimal_curve:
        plt.ylabel('% Optimal Action')
        # plt.title("Optimal action % per step with different epsilon values")
    else:
        plt.ylabel('Average reward')
        # plt.title("Average performance of $\epsilon$ greedy method on 10 arm testbed")
    plt.xlabel('Steps')

    plt.legend()
    plt.show()


def experiment(param_list, stationary=True, optimal_curve_plot=False, number_of_arms=10, runs=200,
               steps=1000):
    total_rewards = []
    total_optimal_rewards = []
    optimal_value = 0
    for i in range(runs):
        if not stationary:
            true_values = [0 for i in range(number_of_arms)]
        else:
            true_values = np.random.normal(0, 1, number_of_arms)
        max_value = max(true_values)
        optimal_value += max_value
        B = Bandit()
        rewards_per_trial = []
        optimal_rewards_per_trial = []
        for j in true_values:
            if j == max_value:
                B.add_action(Action(j, 0), True)
            else:
                B.add_action(Action(j, 0), False)
        for params in range(len(param_list)):
            B.reset(true_values, param_list[params].initial_values)
            rewards_per_eps, optimal_rewards_per_eps = run_bandit_trial(B, param_list[params].epsilon, param_list[params].ucb, param_list[params].confidence_level, steps, stationary,
                                                                        optimal_curve_plot, param_list[params].sample_average, param_list[params].alpha)
            rewards_per_trial.append(rewards_per_eps)
            optimal_rewards_per_trial.append(optimal_rewards_per_eps)
        total_rewards.append(rewards_per_trial)
        total_optimal_rewards.append(optimal_rewards_per_trial)

    average_blocks = []
    average_optimal_blocks = []
    for i in range(len(param_list)):
        eps_block = []
        eps_optimal = []
        for j in range(runs):
            eps_block.append(total_rewards[j][i])
            eps_optimal.append(total_optimal_rewards[j][i])
        average_blocks.append(eps_block)
        average_optimal_blocks.append(eps_optimal)

    plot_curve(runs, param_list, steps, average_blocks, average_optimal_blocks, optimal_curve_plot)


def q6(optimal_curve_plot, number_of_arms, runs, steps):
    P1 = Params(0, 1, False, True, [0 for i in range(number_of_arms)])
    P2 = Params(0.1, 1, False, True, [0 for i in range(number_of_arms)])
    P3 = Params(0.01, 1, False, True, [0 for i in range(number_of_arms)])
    experiment([P1, P2, P3], True, optimal_curve_plot, number_of_arms, runs, steps)


def q7(optimal_curve_plot, number_of_arms, runs, steps):
    P1 = Params(0.1, 1, False, True, [0 for i in range(number_of_arms)])
    P2 = Params(0.1, 0.1, False, False, [0 for i in range(number_of_arms)])
    experiment([P1, P2], False, optimal_curve_plot, number_of_arms, runs, steps)

def q8(optimal_curve_plot, number_of_arms, runs, steps):
    P1 = Params(0, 1, False, True, [5 for i in range(number_of_arms)])
    P2 = Params(0.1, 1, False, True, [0 for i in range(number_of_arms)])
    P3 = Params(0.1, 1, False, True, [5 for i in range(number_of_arms)])
    P4 = Params(0, 1, False, True, [0 for i in range(number_of_arms)])
    experiment([P1, P2, P3, P4], True, optimal_curve_plot, number_of_arms, runs, steps)
    # P5 = Params(0, 1, True, True, [0 for i in range(number_of_arms)])
    # P6 = Params(0.1, 1, False, True, [0 for i in range(number_of_arms)])
    # experiment([P5, P6], True, optimal_curve_plot, number_of_arms, runs, steps)

if __name__ == "__main__":
    number_of_arms = 10
    runs = 2000
    steps = 10000
    optimal_curve_plot = False
    # q4(10,10000)
    # q6(optimal_curve_plot, number_of_arms, runs, steps)
    # q7(optimal_curve_plot, number_of_arms, runs, steps)
    # q8(optimal_curve_plot, number_of_arms, runs, steps)
