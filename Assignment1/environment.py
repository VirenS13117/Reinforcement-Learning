from random import seed
from random import randint, random


seed(1)

from Assignment1 import State


class Environment:
    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.blocks = [(0, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 0), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 9),
                       (5, 10), (6, 4), (7, 4), (8, 4), (10, 4)]
        self.lower_limit = self.left_limit = 0
        self.upper_limit = self.right_limit = 10
        self.prob = 0.8

    def is_target(self, current_state):
        print("checking target")
        print(current_state.x, " ", current_state.y)
        print("target state")
        print(self.target[0], " ", self.target[1])
        if current_state.x == self.target[0] and current_state.y == self.target[1]:
            print("target reached")
            return True
        else:
            return False
