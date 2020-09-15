# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# generate random integer values
from random import seed
from random import randint, random

# seed random number generator
import q3

seed(1)


class Action:
    def __init__(self, action):
        self.action = action
        self.action_list = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

    def get_left_perpendicular(self):
        if self.action == "up":
            return self.action_list["left"]
        elif self.action == "left":
            return self.action_list["down"]
        elif self.action == "down":
            return self.action_list["right"]
        else:
            return self.action_list["up"]

    def get_right_perpendicular(self):
        if self.action == "up":
            return self.action_list["right"]
        elif self.action == "left":
            return self.action_list["up"]
        elif self.action == "down":
            return self.action_list["left"]
        else:
            return self.action_list["down"]

    def get_opposite(self):
        if self.action == "up":
            return self.action_list["down"]
        elif self.action == "left":
            return self.action_list["right"]
        elif self.action == "down":
            return self.action_list["up"]
        else:
            return self.action_list["left"]


class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self, action):
        self.x += action[0]
        self.y += action[1]


class Environment:
    def __init__(self, source, target):
        self.source = State(source[0], source[1])
        self.target = State(target[0], target[1])
        self.blocks = [(0, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 0), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 9),
                       (5, 10), (6, 4), (7, 4), (8, 4), (10, 4)]
        self.lower_limit = self.left_limit = 0
        self.upper_limit = self.right_limit = 10
        self.prob = 0.8

    def out_of_bounds(self, current_state):
        return current_state.x < self.left_limit or current_state.x > self.right_limit \
               or current_state.y < self.lower_limit or current_state.y > self.upper_limit \
               or (current_state.x, current_state.y) in self.blocks

    def is_target(self, current_state):
        print("checking target")
        print(current_state.x, " ", current_state.y)
        print("target state")
        print(self.target.x, " ", self.target.y)
        if current_state.x == self.target.x and current_state.y == self.target.y:
            return True
        else:
            return False


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Create Grid")
    # q1.play_simulation()
    q3.cumulative_reward()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
