class WorsePolicy:
    def __init__(self, grid):
        self.name = "Worse"
        height = grid.upper_limit - grid.lower_limit + 1
        width = grid.right_limit - grid.left_limit + 1
        self.lookup = [[{"up": 0, "down": 0, "left": 0, "right": 0}] * height] * width
        return

    def get_action(self, curr_state):
        return "left"
