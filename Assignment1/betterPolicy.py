from random import random, choice


class BetterPolicy:
    def __init__(self, target):
        self.name = "Better"
        self.lookup = [[{"up": 0, "down": 0, "left": 0, "right": 0}] * 11] * 11
        self.target = target
        return

    def get_action(self, curr_state):
        action_prob = random()
        if action_prob < 0.7:
            return choice(["up", "down", "left", "right"])
        if curr_state.x<self.target[0]:
            if curr_state.y<self.target[1]:
                return choice(["up", "right"])
            else:
                return choice(["down", "right"])
        else:
            if curr_state.y<self.target[1]:
                return choice(["up", "left"])
            else:
                return choice(["down", "left"])
        return choice(["up", "down", "left", "right"])
    