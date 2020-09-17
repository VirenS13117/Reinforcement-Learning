from Assignment1 import Action, State
from random import random

import matplotlib.pyplot as plt


class Simulation:
    def __init__(self, grid, policy):
        self.grid = grid
        self.policy = policy

    def plot_cumulative_rewards(self, trial_rewards):
        average_bar = []
        for i in range(len(trial_rewards[0])):
            sum_rewards = 0
            for j in range(10):
                sum_rewards += trial_rewards[j][i]
            average_bar.append(sum_rewards / len(trial_rewards))

        for i in trial_rewards:
            plt.plot(i, linestyle='dashed')

        plt.plot(average_bar, linewidth=2, label=self.policy.name)
        plt.ylabel('Cumulative reward')
        plt.xlabel('Steps')
        plt.legend()
        plt.show()

    def transition(self, current_state, action):
        print("simulating")
        decision_prob = random()
        print("decision probability : ", decision_prob)
        if decision_prob < self.grid.prob:
            current_state.move(action.action_list[action.action])
        elif self.grid.prob < decision_prob < self.grid.prob + (1 - self.grid.prob) / 2:
            current_state.move(action.get_left_perpendicular())
        else:
            current_state.move(action.get_right_perpendicular())
        reward = 0
        if self.grid.is_target(current_state):
            current_state.__setstate__(self.grid.source[0], self.grid.source[1])
            reward = 1
        return current_state, reward

    def simulate(self, num_trials=10, num_steps=10000):
        print("Q3 : Random Policy")
        trial_rewards = []
        for trials in range(num_trials):
            curr_state = State.State(self.grid.source[0], self.grid.source[1], self.grid)
            steps_reward = []
            cum_sum = 0
            for steps in range(num_steps):
                action = self.policy.get_action(curr_state)
                action_object = Action.Action(action)
                curr_state, reward = self.transition(curr_state, action_object)
                cum_sum += reward
                steps_reward.append(cum_sum)
            trial_rewards.append(steps_reward)
        self.plot_cumulative_rewards(trial_rewards)
