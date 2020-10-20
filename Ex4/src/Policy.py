# from  Ex4.src. import GridWorld
import matplotlib.pyplot as plt


class Policy:
    def __init__(self, name):
        self.name = name
        return

    def get_action(self, observation):
        return observation < 20
