from  Ex3.src.GridWorld import GridWorld
import matplotlib.pyplot as plt
class Policy:
    def __init__(self, states, actions):
        self.states = states.copy()
        self.actions = actions.copy()
        self.default_prob = 1/len(actions)
        self.map = dict()
        return

    def get_probability(self, state, action):
        if (state in self.states) and (action in self.actions):
            return self.map.get((state, action), self.default_prob)
        else:
            return 0

    def policy_map(self, state, action, prob):
        self.map[(state, action)] = prob
        return

    def plot(self, gw):
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
        x_pos = []
        y_pos = []
        x_dir = []
        y_dir = []
        for rs in range(gw.rows):
            for cs in range(gw.cols):
                current_state = gw.get_state_id((rs, cs))
                for action in gw.get_actions():
                    if self.get_probability(current_state, action):
                        y_pos.append(rs + 0.5)
                        x_pos.append(cs + 0.5)
                        move = gw.make_movement(action)
                        x_dir.append(move[1])
                        y_dir.append(move[0] * -1)
        ax.quiver(x_pos, y_pos, x_dir, y_dir)
        xticks = [i for i in range(gw.cols + 1)]
        ax.set_xlim(0, gw.cols)
        ax.set_xticks(xticks)
        yticks = [i for i in range(gw.rows + 1)]
        ax.set_ylim(gw.rows, 0)
        ax.set_yticks(yticks[::-1])
        ax.grid()
        ax.set_title("Optimal Policy $\pi_{*}$")
        plt.show()
        return