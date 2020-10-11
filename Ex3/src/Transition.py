from Ex3.src.Action import Action
class Transition:
    def __init__(self, actions, default_reward=0):
        self.map = dict()
        self.actions = actions
        self.default_reward = default_reward
        return

    def transition_map(self, state, action, next_state, reward, probability=1):
        if self.actions.is_valid_action(action):
            self.map[(state, action)] = (next_state, reward, probability)
        return

    def next(self, state, action):
        return self.map.get((state,action), (state, self.default_reward, 1))
