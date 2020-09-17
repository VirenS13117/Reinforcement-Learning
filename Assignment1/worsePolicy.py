class WorsePolicy:
    def __init__(self):
        self.name = "Worse"
        return

    def get_action(self, curr_state):
        return "left"
