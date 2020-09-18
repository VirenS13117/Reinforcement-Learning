import random

class RandomPolicy:
    def __init__(self, grid):
        self.name = "Random"
        height = grid.upper_limit - grid.lower_limit + 1
        width = grid.right_limit - grid.left_limit + 1
        self.lookup = [[{"up": 0, "down": 0, "left": 0, "right": 0}] * height] * width

    def get_action(self, curr_state):
        return random.choice(["up", "down", "left", "right"])

    def print_lookup(self):
        for i in range(len(self.lookup)):
            for j in range(len(self.lookup[0])):
                print(i, " ", j, " ", self.lookup[i][j])

