import random

class RandomPolicy:
    def __init__(self):
        return

    def get_action(self):
        return random.choice(["up", "left", "down", "right"])

