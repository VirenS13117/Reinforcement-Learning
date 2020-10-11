class Action:
    def __init__(self, action_directions):
        self.action_directions = action_directions.copy()
        self.actions = []
        for action_name, action_step in action_directions.items():
            self.actions.append(action_name)
        return

    def is_valid_action(self, action):
        return action in self.actions

    def get_direction(self, action):
        return self.action_directions.get(action, (0,0))

    def get_all_actions(self):
        return self.actions.copy()

