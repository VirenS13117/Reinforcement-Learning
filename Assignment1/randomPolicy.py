import random

class RandomPolicy:
    def __init__(self):
        self.name = "Random"
        return

    def get_action(self, curr_state):
        return random.choice(["up", "left", "down", "right"])

