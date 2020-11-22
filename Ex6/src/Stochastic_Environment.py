import numpy as np
class StochasticGrid:
    def __init__(self, blocks):
        self.min_x = 0
        self.max_x = 8
        self.min_y = 0
        self.max_y = 5
        self.start = (3,0)
        self.goal_state = (8,5)
        self.actions = ["left", "right", "up", "down"]
        self.curr_state = self.start
        self.blocks = blocks
        return

    def change_blocklist(self, new_blocks):
        self.blocks = new_blocks
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
        else:
            print("invalid action id")
            return -1

    def isLegalState(self, curr_state):
        return (curr_state not in self.blocks) and (self.min_x <= curr_state[0] <= self.max_x) and (self.min_y <= curr_state[1] <= self.max_y)

    def make_transition(self, state, action):
        reward = 0
        done = False
        curr_state = self.make_move(state, self.get_action_name(action))
        if self.is_goal(curr_state):
            reward = 1
            done = True
        return curr_state, reward, done, {}

    def left_perpendicular(self, action):
        if action == "up":
            return "left"
        elif action == "left":
            return "down"
        elif action == "down":
            return "right"
        else:
            return "up"

    def right_perpendicular(self, action):
        if action == "up":
            return "right"
        elif action == "left":
            return "up"
        elif action == "down":
            return "left"
        else:
            return "down"

    def get_deterministic_action(self, action):
        num = np.random.random()
        action_left = self.left_perpendicular(action)
        action_right = self.right_perpendicular(action)
        return num,[(action,0.8), (action_left, 0.1), (action_right, 0.1)]

    def step(self, action):
        done = False
        epsilon, actions_list = self.get_deterministic_action(self.get_action_name(action))
        optimal_state = self.make_move(self.curr_state, actions_list[0][0])
        state_left = self.make_move(self.curr_state, actions_list[1][0])
        state_right = self.make_move(self.curr_state, actions_list[2][0])
        reward_optimal, reward_left, reward_right = 0, 0, 0
        if self.is_goal(optimal_state):
            reward_optimal = 1
        if self.is_goal(state_left):
            reward_left = 1
        if self.is_goal(state_right):
            reward_right = 1
        self.curr_state = optimal_state
        info = [(optimal_state, reward_optimal, 0.8), (state_left, reward_left, 0.1), (state_right, reward_right, 0.1)]
        return self.curr_state, reward_optimal, done, info

    def get_current_state(self):
        return self.curr_state

    def make_move(self, state, action):
        dx, dy = 0, 0
        new_state = state
        if action == "up":
            dy += 1
        elif action == "down":
            dy += -1
        elif action == "left":
            dx += -1
        elif action == "right":
            dx += 1
        else:
            print("wrong action : ", action)
            return new_state
        new_state = (state[0]+dx, state[1]+dy)
        if self.isLegalState(new_state):
            return new_state
        return state
