
class ManualPolicy:
    def __init__(self):
        self.name = "Manual"
        self.lookup = [[{"up": 0, "down": 0, "left": 0, "right": 0}] * 11] * 11
        return

    def get_action(self, curr_state):
        print("input action")
        action_input = input()
        return action_input

