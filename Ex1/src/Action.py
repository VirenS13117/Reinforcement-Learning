import numpy as np


class Action:
    def __init__(self, value, initial_value, alpha):
        self.true_value = value
        self.occurrence = 1
        self.current_value = initial_value
        self.alpha = alpha

    def play_action(self):
        return self.true_value + np.random.normal()

    def sample_average_update(self, reward):
        self.occurrence += 1
        self.current_value += (reward - self.current_value) / self.occurrence


    def exponential_recency_weighted_average(self, reward):
        self.current_value = self.current_value + self.alpha(reward - self.current_value)