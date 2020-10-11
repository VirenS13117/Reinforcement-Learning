import numpy as np
from Ex3.src.Action import Action
from Ex3.src.Transition import Transition


class GridWorld:
    def __init__(self, num_rows, num_cols, actions, dr=0, oob_r=0, sr=None):
        self.rows = num_rows
        self.cols = num_cols
        self.states = np.array([[cols + rows * self.rows for cols in range(self.cols)] for rows in range(self.cols)])
        self.grid = np.array([[(rows, cols) for cols in range(self.cols)] for rows in range(self.cols)])
        self.actions = Action(actions)
        self.transitions = Transition(self.actions)
        self.move_next(dr, oob_r, sr)
        return

    def get_state_id(self, state_coordinates):
        if state_coordinates in self.grid:
            return self.states[state_coordinates[0]][state_coordinates[1]]
        return -1

    def get_state_position(self, state_id):
        if state_id in self.states:
            row = int(state_id / self.cols)
            col = int(state_id % self.cols)
            return self.grid[row][col]
        else:
            return (-1, -1)

    def move_next(self, dr, oob_r, sr):
        available_actions = self.actions.get_all_actions()
        for row in range(self.rows):
            for col in range(self.cols):
                for each_action in available_actions:
                    curr_state = self.get_state_id((row, col))
                    if sr and (curr_state in sr):
                        next_state, reward = sr.get(curr_state)
                        self.transitions.transition_map(curr_state, each_action, next_state, reward)
                    else:
                        dir = self.actions.get_direction(each_action)
                        next_x = row + dir[0]
                        next_y = col + dir[1]
                        if (0 <= next_x < self.rows) and (0 <= next_y < self.cols):
                            next_state = self.get_state_id((next_x, next_y))
                            self.transitions.transition_map(curr_state, each_action, next_state, dr)
                        else:
                            self.transitions.transition_map(curr_state, each_action, curr_state, oob_r)
        return

    def make_transition(self, state, action):
        return self.transitions.next(state, action)

    def make_movement(self, action):
        return self.actions.get_direction(action)

    def get_states(self):
        return self.states

    def get_actions(self):
        return self.actions.get_all_actions()