import random


class LearnedPolicy:
    def __init__(self):
        self.name = "Learned"
        self.lookup = [[{"up": 0, "down": 0, "left": 0, "right": 0}] * 11] * 11
        self.epsilon = 0.1

    def get_action(self, curr_state):
        action_prob = random.random()
        if action_prob > self.epsilon:
            curr_state_actions = self.lookup[curr_state.x][curr_state.y]
            all_values = curr_state_actions.values()
            max_value = max(all_values)
            action_list = []
            for i in curr_state_actions:
                if curr_state_actions[i] == max_value:
                    action_list.append(i)
            best_action = random.choice(action_list)
            print("best action according to learned policy : ", best_action)
            return best_action
        return random.choice(["up", "down", "left", "right"])

    def get_max_value(self, curr_state):
        curr_state_actions = self.lookup[curr_state.x][curr_state.y]
        all_values = curr_state_actions.values()
        max_value = max(all_values)
        return max_value

    def print_lookup(self):
        for i in range(len(self.lookup)):
            for j in range(len(self.lookup[0])):
                print(i, " ", j, " ", self.lookup[i][j])
