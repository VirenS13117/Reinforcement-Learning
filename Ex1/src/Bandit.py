import random
import numpy as np


class Bandit:
    def __init__(self):
        self.actions = []
        self.optimal_action = 0

    def reset(self, true_values, initial_values):
        for i in range(len(true_values)):
            self.actions[i].current_value = 0
            self.actions[i].occurrence = 0
            self.actions[i].true_value = true_values[i]
            self.actions[i].initial_value = initial_values[i]
        return

    def update(self):
        for action in self.actions:
            action.true_value += np.random.normal(0, 0.01)
            if action.true_value > self.optimal_action.true_value:
                self.optimal_action = action
        return

    def add_action(self, action, is_optimal=False):
        self.actions.append(action)
        if is_optimal:
            self.optimal_action = action
        return

    def choose_action(self, epsilon):
        val = random.random()
        if val <= epsilon:
            return random.choice(self.actions)
        else:
            all_values = [i.current_value for i in self.actions]
            max_value = max(all_values)
            action_list = []
            for i in self.actions:
                if i.current_value == max_value:
                    action_list.append(i)
            best_action = random.choice(action_list)
            return best_action
