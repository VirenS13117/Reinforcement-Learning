
class State:
    def __init__(self, x, y, grid):
        self.x = x
        self.y = y
        self.grid = grid

    def out_of_bounds(self, next_x, next_y):
        return next_x < self.grid.left_limit or next_x > self.grid.right_limit \
               or next_y < self.grid.lower_limit or next_y > self.grid.upper_limit \
               or (next_x, next_y) in self.grid.blocks

    def move(self, action):
        next_x = self.x + action[0]
        next_y = self.y + action[1]

        if not self.out_of_bounds(next_x, next_y):
            self.x += action[0]
            self.y += action[1]

    def __setstate__(self, x, y):
        self.x = x
        self.y = y