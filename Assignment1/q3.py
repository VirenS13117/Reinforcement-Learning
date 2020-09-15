import random
from Assignment1 import q1
from main import Action, State, Environment
import matplotlib.pyplot as plt


def random_policy():
    return random.choice(["up", "left", "down", "right"])


def plot_cumulative_rewards(trial_rewards):
    average_bar = []
    for i in range(10000):
        sum_rewards = 0
        for j in range(10):
            sum_rewards += trial_rewards[j][i]
        average_bar.append(sum_rewards/10)

    for i in trial_rewards:
        plt.plot(i, linestyle='dashed')

    plt.plot(average_bar)
    plt.ylabel('Cumulative reward')
    plt.xlabel('Steps')
    plt.show()


def cumulative_reward(iterations=10):
    source = (0, 0)
    target = (10, 10)
    my_grid = Environment(source, target)
    print("Q3 : Random Policy")
    trial_rewards = []
    for trials in range(10):
        curr_state = State(source[0], source[1])
        steps_reward = []
        cum_sum = 0
        for steps in range(10000):
            action = random_policy()
            action_object = Action(action)
            curr_state.move(action_object.action_list[action])
            if my_grid.out_of_bounds(curr_state):
                print("out of bound")
                curr_state.move(action_object.get_opposite())
            reward = 0
            if my_grid.is_target(curr_state):
                reward = 1
            cum_sum += reward
            steps_reward.append(cum_sum)
        trial_rewards.append(steps_reward)
    plot_cumulative_rewards(trial_rewards)
