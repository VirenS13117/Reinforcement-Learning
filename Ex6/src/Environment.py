
class Grid:
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
        if self.isLegalAction(new_state):
            return new_state
        return state
