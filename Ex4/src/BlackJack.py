import gym
import numpy as np
from collections import defaultdict
from Ex4.src.Policy import Policy
from Ex4.src.plot_results import plot_value_fn, plot_policy
from Ex4.src.randomPolicy import RandomPolicy
from typing import List, Tuple, Dict, Set, Callable

def get_average_return(returns):
    print(returns)
    return np.average(returns)

def first_visit_monte_carlo_policy_evaluation(env, policy, num_episodes = 100, gamma=1):
    value_function = defaultdict(float)
    count_occurence = defaultdict(float)
    for i in range(num_episodes):
        state = env.reset()
        episode = []
        isTerminal = False
        while not isTerminal:
            action = policy.get_action(state)
            next_state, reward, isTerminal, info = env.step(action)
            episode.append((state,action,reward))
            state = next_state
        G = 0
        for i,(state, action, reward) in enumerate(episode[::-1]):
            G = gamma*G + reward
            previous_states = [j[0] for j in episode[0:len(episode)-i-1]]
            if state not in previous_states:
                count_occurence[state] += 1
                value_function[state] = (value_function[state]*(count_occurence[state]-1) + G)/(count_occurence[state])
    env.close()
    return value_function

def monte_carlo_exploring_starts(env, policy, num_episodes=100, gamma=1):
    value_function = defaultdict(float)
    print("env action n : ", env.action_space.n)
    count_occurence = defaultdict(lambda: np.zeros(env.action_space.n, dtype=np.float))
    q_value = defaultdict(lambda: np.zeros(env.action_space.n, dtype=np.float))
    pi = {}#defaultdict(float)
    for i in range(num_episodes):
        state = env.reset()
        action = env.action_space.sample()
        episode = []
        isTerminal = False
        while not isTerminal:
            next_state, reward, isTerminal, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            action = pi.get(state, policy.get_action(state))
        G = 0
        for i,(state, action, reward) in enumerate(episode[::-1]):
            G = gamma*G + reward
            previous_states = [(j[0],j[1]) for j in episode[0:len(episode)-i-1]]
            if (state,action) not in previous_states:
                count_occurence[state][action] += 1
                q_value[state][action] = (q_value[state][action]*(count_occurence[state][action]-1)+G)/(count_occurence[state][action])
                pi[state] = np.random.choice(np.where((q_value[state] == np.max(q_value[state])))[0])
    env.close()
    for state, actions in q_value.items():
        value_function[state] = np.max(actions)

    return value_function, pi

def part_a():
    policy = Policy("sticks on 20-21")
    env = gym.make("Blackjack-v0")
    value_function = first_visit_monte_carlo_policy_evaluation(env, policy, 10000)
    plot_value_fn(value_function, "After 10,000 episodes")
    env = gym.make("Blackjack-v0")
    value_function = first_visit_monte_carlo_policy_evaluation(env, policy, 500000)
    plot_value_fn(value_function, "After 500,000 episodes")

def part_b():
    # policy = RandomPolicy("randomPolicy")
    policy = Policy("sticks on 20-21")
    env = gym.make("Blackjack-v0")
    value_function, pi = monte_carlo_exploring_starts(env, policy, 500000)
    # plot_value_fn(value_function, "After 500,000 episodes")
    plot_policy(pi, policy, "$\pi_{*}$")



if __name__ == "__main__":

    ##
    # part_a()
    ##
    part_b()

