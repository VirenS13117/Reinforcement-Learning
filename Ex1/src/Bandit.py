import random
import numpy as np


class Bandit:
    def __init__(self):
        self.actions = []
        self.optimal_action = 0

    def reset(self):
        for action in self.actions:
            action.current_value = 0
            action.occurrence = 0
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
