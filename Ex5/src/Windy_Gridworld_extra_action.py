import numpy as np
from typing import Tuple, List, Set, Dict, Callable
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from tqdm import trange


class extra_move_Grid:
    def __init__(self):
        self.min_x = 0
        self.max_x = 9
        self.min_y = 0
        self.max_y = 6
        self.start = (0, 3)
        self.goal_state = (7, 3)
        self.actions = ["left", "right", "up", "down", "north_east", "south_east", "south_west", "north_west", "no_move"]
        self.wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.curr_state = self.start
        return

    def is_goal(self, state):
        return state[0] == self.goal_state[0] and state[1] == self.goal_state[1]

    def reset(self):
        self.curr_state = self.start
        return self.curr_state

    def get_action_name(self, action_id):
        if action_id == 0:
            return "left"
        elif action_id == 1:
            return "right"
        elif action_id == 2:
            return "up"
        elif action_id == 3:
            return "down"
        elif action_id == 4:
            return "north_east"
        elif action_id == 5:
            return "south_east"
        elif action_id == 6:
            return "south_west"
        elif action_id == 7:
            return "north_west"
        elif action_id == 8:
            return "no_move"
        else:
            print("invalid action id")
            return -1

    def step(self, action):
        reward = -1
        done = False
        self.curr_state = self.make_move(self.get_action_name(action))
        if self.is_goal(self.curr_state):
            reward = 0
            done = True
        return self.curr_state, reward, done, {}

    def get_current_state(self):
        return self.curr_state

    def get_wind_strength(self, state):
        return self.wind_strength[state[0]]

    def make_move(self, action):
        state = self.curr_state
        dx, dy = 0, 0
        wind_shift = self.wind_strength[state[0]]
        new_state = state
        if action == "up":
            dy += 1
        elif action == "down":
            dy += -1
        elif action == "left":
            dx += -1
        elif action == "right":
            dx += 1
        elif action == "north_east":
            dy += 1
            dx += 1
        elif action == "south_east":
            dx += 1
            dy += -1
        elif action == "south_west":
            dx += -1
            dy += -1
        elif action == "north_west":
            dx += -1
            dy += 1
        elif action == "no_move":
            dx, dy = 0, 0
        else:
            print("wrong action : ", action)
            return new_state
        if self.min_x <= state[0] + dx <= self.max_x:
            if self.min_y <= state[1] + dy + wind_shift <= self.max_y:
                new_state = (state[0] + dx, state[1] + dy + wind_shift)
            elif self.min_y <= state[1] + dy <= self.max_y:
                new_state = (state[0] + dx, state[1] + dy)
        return new_state
