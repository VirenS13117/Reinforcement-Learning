
class ManualPolicy:
    def __init__(self):
        self.name = "Manual"
        return

    def get_action(self, curr_state):
        print("input action")
        action_input = input()
        return action_input

