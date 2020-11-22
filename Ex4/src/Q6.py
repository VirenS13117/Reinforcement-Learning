from typing import Tuple, List, Set, Dict, Callable
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from Ex4.src.plot_results import plot_performance, plot_policy_values
from tqdm import trange


class Grid:
    def __init__(self, target):
        self.start = (0, 0)
        self.state = self.start
        self.end = (10, 10)
        self.target = target
        self.actions = ["left", "right", "up", "down"]
        self.blocks = [(0, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 0), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 9),
                       (5, 10), (6, 4), (7, 4), (8, 4), (10, 4)]
        self.direction = {"left": (-1, 0), "right": (1, 0), "up": (0, 1), "down": (0, -1)}
        self.states = []
        for x in range(self.start[0], self.end[0] + 1):
            for y in range(self.start[1], self.end[1] + 1):
                self.states.append((x, y))

    def get_action_index(self, action):
        if action == "left":
            return 0
        elif action == "right":
            return 1
        elif action == "up":
            return 2
        elif action == "down":
            return 3
        else:
            print("invalid action")
            return -1

    def get_action_name(self, action_id):
        if action_id == 0:
            return "left"
        elif action_id == 1:
            return "right"
        elif action_id == 2:
            return "up"
        elif action_id == 3:
            return "down"
        else:
            print("invalid action id")
            return -1

    def is_block(self, state):
        if state in self.blocks:
            return True
        return False

    def is_out_of_bounds(self, state):
        return state[0] < self.start[0] or state[0] > self.end[0] or state[1] < self.start[1] or state[1] > self.end[1]

    def is_goal(self, state):
        return state[0] == self.target[0] and state[1] == self.target[1]

    def left_perpendicular(self, action):
        if action == "up":
            return "left"
        elif action == "left":
            return "down"
        elif action == "down":
            return "right"
        else:
            return "up"

    def right_perpendicular(self, action):
        if action == "up":
            return "right"
        elif action == "left":
            return "up"
        elif action == "down":
            return "left"
        else:
            return "down"

    def get_deterministic_action(self, action):
        num = np.random.random()
        if num < 0.8:
            return action
        if num < 0.9:
            return self.left_perpendicular(action)
        return self.right_perpendicular(action)

    def make_move(self, state, action):
        x, y = self.direction[action]
        new_state = (state[0] + x, state[1] + y)
        if self.is_block(new_state) or self.is_out_of_bounds(new_state):
            return state
        return new_state

    def reset(self, random_state):
        if random_state:
            x = np.random.choice(np.arange(self.start[0], self.end[0] + 1))
            y = np.random.choice(np.arange(self.start[1], self.end[1] + 1))
            while self.is_block((x, y)) or self.is_out_of_bounds((x, y)):
                x = np.random.choice(np.arange(self.start[0], self.end[0] + 1))
                y = np.random.choice(np.arange(self.start[1], self.end[1] + 1))
            self.state = (x, y)
        else:
            self.state = self.start
        return self.state

    def step(self, action):
        reward = 0
        done = False
        if action in self.actions:
            valid_action = self.get_deterministic_action(action)
            new_state = self.make_move(self.state, valid_action)
            self.state = new_state
            if self.is_goal(new_state):
                reward = 1
                done = True
            return new_state, reward, done, {}
        return self.state, 0, False, {}


def create_env(goal_state):
    env = Grid(goal_state)
    if env.is_block(goal_state) or env.is_out_of_bounds(goal_state):
        print("Not a valid goal state")
        return None

    return env


def policy_action(state, qvalues, epsilon):
    num = np.random.random()
    if num <= epsilon:
        return np.random.choice(np.arange(len(qvalues[state])))
    return np.random.choice(np.where(qvalues[state] == np.max(qvalues[state]))[0])


def monte_carlo_policy_control(env, num_trials, num_episodes, T, epsilon=0.1, gamma=0.99):
    log_trials = []
    for n in trange(num_trials, desc="trial", leave=False):
        number_of_actions = len(env.actions)
        count_occurence = defaultdict(lambda: np.zeros(number_of_actions, dtype=np.float))
        q_value = defaultdict(lambda: np.zeros(number_of_actions, dtype=np.float))
        q_policy = defaultdict(
            lambda: np.array([(1.0 / float(number_of_actions)) for k in np.arange(number_of_actions)]))
        log_episodes = []
        for e in trange(num_episodes):
            state = env.reset(False)
            episode = []
            isTerminal = False
            count = 0
            while not isTerminal and count < T:
                action = policy_action(state, q_policy, epsilon)
                next_state, reward, isTerminal, info = env.step(env.get_action_name(action))
                episode.append((state, action, reward))
                state = next_state
                count += 1
            log_episodes.append(episode)
            G = 0
            for i, (state, action, reward) in enumerate(episode[::-1]):
                G = gamma * G + reward
                previous_states = [(j[0], j[1]) for j in episode[0:len(episode) - i - 1]]
                if (state, action) not in previous_states:
                    count_occurence[state][action] += 1
                    q_value[state][action] = (q_value[state][action] * (count_occurence[state][action] - 1) + G) / (
                        count_occurence[state][action])
                    a_max = np.random.choice(np.where((q_value[state] == np.max(q_value[state])))[0])
                    for a in np.arange(number_of_actions):
                        q_policy[state][a] = ((1 - epsilon) + epsilon / float(number_of_actions)) if a == a_max else (
                                    epsilon / float(number_of_actions))
        log_trials.append(log_episodes)
    return np.array(log_trials), q_value, q_policy


def monte_carlo_prediction(env, num_episodes, T, target_policy, behavior_policy, log_trials, offPolicy=False,
                           epsilon=0.1, gamma=0.99):
    number_of_actions = len(env.actions)
    count_occurence = defaultdict(lambda: np.zeros(number_of_actions, dtype=np.float))
    q_value = defaultdict(lambda: np.random.random(number_of_actions))
    c_sum = defaultdict(lambda: np.zeros(number_of_actions, dtype=np.float))
    for e in trange(num_episodes):
        if offPolicy:
            episode = log_trials[0][e]
        else:
            episode = []
            state = env.reset(False)
            isTerminal = False
            count = 0
            while not isTerminal and count < T:
                action = target_policy[state]
                next_state, reward, isTerminal, info = env.step(env.get_action_name(action))
                episode.append((state, action, reward))
                state = next_state
                count += 1
        G = 0
        W = 1
        for i, (state, action, reward) in enumerate(episode[::-1]):
            G = gamma * G + reward
            c_sum[state][action] += W
            count_occurence[state][action] += 1
            q_value[state][action] = (q_value[state][action] * (count_occurence[state][action] - 1) + G) / (
            count_occurence[state][action])
            if offPolicy:
                W *= (float(1 if action == target_policy[state] else 0) / behavior_policy[state][action])
            if W == 0:
                break
    return q_value


def compute_greedy(env, q_values):
    q_policy = defaultdict(lambda: np.random.choice(np.arange(len(env.actions()))))
    for state in env.states:
        q_policy[state] = np.random.choice(np.where((q_values[state] == np.max(q_values[state])))[0])
    return q_policy


def q_b(env, trial_data, behavioral_value, behavioral_policy):
    greedy_policy = compute_greedy(env, behavioral_value)
    plot_policy_values(env, greedy_policy, title="$\pi_{greedy}$ : Greedy Policy")
    plot_policy_values(env, greedy_policy, Q=behavioral_value, title="$Q_{\pi}$ : Q values of the greedy policy")
    return greedy_policy


def q_c(env, trial_data, behavioral_value, behavioral_policy, greedy_policy):
    q_value = monte_carlo_prediction(env, trial_data.shape[1], 459, greedy_policy, behavioral_policy, trial_data,
                                     offPolicy=True)
    q_greedy = compute_greedy(env, behavioral_value)
    plot_policy_values(env, q_greedy, Q=q_value, title="Off Policy")
    return

def q_d(env, trial_data, behavioral_value, behavioral_policy, greedy_policy):
    q_value = monte_carlo_prediction(env, trial_data.shape[1], 459, greedy_policy, behavioral_policy, trial_data,
                                     offPolicy=False)
    q_greedy = compute_greedy(env, behavioral_value)
    plot_policy_values(env, q_greedy, Q=q_value, title="On Policy")
    return

if __name__ == "__main__":
    goal_state = (10, 10)
    env = create_env(goal_state)
    trial_data, behavioral_value, behavioral_policy = monte_carlo_policy_control(env, num_trials=1, num_episodes=10000,
                                                                                 T=459, epsilon=0.1)
    greedy_policy = q_b(env, trial_data, behavioral_value, behavioral_policy)
    q_c(env, trial_data, behavioral_value, behavioral_policy, greedy_policy)
    q_d(env, trial_data, behavioral_value, behavioral_policy, greedy_policy)
