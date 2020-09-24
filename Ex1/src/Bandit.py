import random


class Bandit:
    def __init__(self, epsilon):
        self.actions = []
        self.epsilon = epsilon

    def add_action(self, action):
        self.actions.append(action)

    def choose_action(self):
        val = random.random()
        if val < self.epsilon:
            return random.choice(self.actions)
        else:
            all_values = [i.current_value for i in self.actions]
            max_value = max(all_values)
            action_list = []
            for i in self.actions:
                if i.current_value == max_value:
                    action_list.append(i)
            best_action = random.choice(action_list)
            return best_action
