# from  Ex4.src. import GridWorld
import matplotlib.pyplot as plt
import numpy as np


class RandomPolicy:
    def __init__(self, name):
        self.name = name
        return

    def get_action(self, observation):
        return np.random.choice([0,1])
