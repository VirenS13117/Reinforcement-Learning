import numpy as np


class Action:
    def __init__(self, value, initial_value, alpha):
        self.true_value = value
        self.occurrence = 0
        self.current_value = initial_value
        self.alpha = alpha

    def play_action(self):
        return np.random.normal(self.true_value, 1)

    def sample_average_update(self, reward):
        self.occurrence += 1
        self.current_value = (self.true_value * (self.occurrence - 1) + reward) / self.occurrence

    def exponential_recency_weighted_average(self, reward):
        self.current_value = self.current_value + self.alpha(reward - self.current_value)