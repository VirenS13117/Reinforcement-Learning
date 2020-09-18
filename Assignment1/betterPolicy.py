from random import random, choice


class BetterPolicy:
    def __init__(self, grid):
        self.name = "Better"
        height = grid.upper_limit - grid.lower_limit + 1
        width = grid.right_limit - grid.left_limit + 1
        self.lookup = [[{"up": 0, "down": 0, "left": 0, "right": 0}] * height] * width
        return

    def get_action(self, curr_state):
        action_prob = random()
        if action_prob < 0.8:
            return choice(["up", "down", "left", "right"])
        if curr_state.x < 5:
            if curr_state.y >= 9:
                return "down"
            elif 6 <= curr_state.y < 8:
                return "up"
            elif curr_state.y == 8:
                return "right"
            elif curr_state.y < 4:
                return "up"
            elif curr_state.y == 4:
                if curr_state.x == 0:
                    return "right"
                elif curr_state.x == 1:
                    return "up"
                else:
                    return "left"
            else:
                return "up"
        elif curr_state.x == 5:
            return "right"
        else:
            if curr_state.y <= 2:
                return "up"
            elif curr_state.y == 3:
                if curr_state.x < 8:
                    return "right"
                elif curr_state.x > 8:
                    return "left"
                else:
                    return "up"
            elif curr_state.y == 10:
                return "right"
            else:
                return "up"
        return "up"
