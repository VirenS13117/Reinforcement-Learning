import numpy as np
from typing import Tuple, List, Set, Dict, Callable
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from tqdm import trange
from Ex5.src.plot_results import plot_performance, plot_histogram
from Ex5.src.Windy_Gridworld import Grid
from Ex5.src.Windy_Gridworld_diagonal_actions import all_direction_Grid
from Ex5.src.Windy_Gridworld_extra_action import extra_move_Grid
from Ex5.src.stochastic_windy_gridworld import StochasticGrid


def epsilon_greedy_policy(Q, state, nA, epsilon):
    probs = np.ones(nA) * epsilon / nA
    best_action = np.argmax(Q[state])
    probs[best_action] += 1.0 - epsilon
    return probs

def n_step_SARSA(env, num_episodes, n_step = 3, alpha = 0.5, gamma=1, epsilon=0.1):
    trial_data = [[] for i in range(10)]
    for trials in range(10):
        q_value = defaultdict(lambda: np.zeros(len(env.actions)))
        cum_steps = 0
        for episode in trange(num_episodes):
            state = env.reset()
            probs = epsilon_greedy_policy(q_value, state, len(env.actions), epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            T = float("+inf")
            tau = 0
            n_stores = []
            episode_complete = True
            n_stores.append((state, action, 0))
            for steps in range(10000):
                if steps<T:
                    next_state, reward, done, _ = env.step(action)
                    if done:
                        new_action = -1
                        n_stores.append((next_state, new_action, reward))
                        T = steps+1
                    else:
                        probs = epsilon_greedy_policy(q_value, next_state, len(env.actions), epsilon)
                        new_action = np.random.choice(np.arange(len(probs)), p=probs)
                        n_stores.append((next_state, new_action, reward))
                tau = steps-n_step+1
                if tau>=0:
                    G = 0
                    for i in range(tau+1, min(tau+n_step, T)+1):
                        G += gamma**(i-tau+1) * n_stores[i][2]

                    if tau+n_step<T:
                        s_update = n_stores[tau + n_step]
                        G += gamma**n_step * q_value[s_update[0]][s_update[1]]
                    q_value[n_stores[tau][0]][n_stores[tau][1]] += alpha*(G-q_value[n_stores[tau][0]][n_stores[tau][1]])
                if tau==T-1:
                    cum_steps += steps
                    trial_data[trials].append(cum_steps)
                    episode_complete = False
                    break
                state = next_state
                action = new_action
            if episode_complete==True:
                cum_steps += 10000
                trial_data[trials].append(cum_steps)
    return trial_data


def expected_SARSA(env, num_episodes, alpha=0.5, gamma=1, epsilon=0.1):
    trial_data = [[] for i in range(10)]
    for trials in range(10):
        q_value = defaultdict(lambda: np.zeros(len(env.actions)))
        cum_steps = 0
        for episode in trange(num_episodes):
            state = env.reset()
            probs = epsilon_greedy_policy(q_value, state, len(env.actions), epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            for step in range(10000):
                next_state, reward, done, _ = env.step(action)
                probs = epsilon_greedy_policy(q_value, next_state, len(env.actions), epsilon)
                new_action = np.random.choice(np.arange(len(probs)), p=probs)
                sum_prob = sum([probs[i] * q_value[next_state][i] for i in range(len(env.actions))])
                td_target = reward + gamma * sum_prob
                td_error = td_target - q_value[state][action]
                q_value[state][action] += alpha * td_error
                if done:
                    cum_steps += step
                    trial_data[trials].append(cum_steps)
                    break
                state = next_state
                action = new_action
    return trial_data


def SARSA(env, num_episodes, alpha=0.5, gamma=1, epsilon=0.1):
    trial_data = [[] for i in range(10)]
    for trials in range(10):
        q_value = defaultdict(lambda: np.zeros(len(env.actions)))
        cum_steps = 0
        for episode in trange(num_episodes):
            state = env.reset()
            probs = epsilon_greedy_policy(q_value, state, len(env.actions), epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            for step in range(10000):
                next_state, reward, done, _ = env.step(action)
                probs = epsilon_greedy_policy(q_value, next_state, len(env.actions), epsilon)
                new_action = np.random.choice(np.arange(len(probs)), p=probs)
                td_target = reward + gamma * q_value[next_state][new_action]
                td_error = td_target - q_value[state][action]
                q_value[state][action] += alpha * td_error
                if done:
                    cum_steps += step
                    trial_data[trials].append(cum_steps)
                    break
                state = next_state
                action = new_action
    return trial_data


def Q_learning(env, num_episodes, alpha=0.5, gamma=1, epsilon=0.1):
    trial_data = [[] for i in range(10)]
    for trials in range(10):
        q_value = defaultdict(lambda: np.zeros(len(env.actions)))
        cum_steps = 0
        for episode in trange(num_episodes):
            state = env.reset()
            for step in range(10000):
                probs = epsilon_greedy_policy(q_value, state, len(env.actions), epsilon)
                action = np.random.choice(np.arange(len(probs)), p=probs)
                next_state, reward, done, _ = env.step(action)
                td_target = reward + gamma * np.amax(q_value[next_state])
                td_error = td_target - q_value[state][action]
                q_value[state][action] += alpha * td_error
                if done:
                    cum_steps += step
                    trial_data[trials].append(cum_steps)
                    break
                state = next_state
    return trial_data


def monte_carlo_policy_control(env, num_episodes, gamma=1, epsilon=0.1):
    number_of_actions = len(env.actions)
    count_occurence = defaultdict(lambda: np.zeros(number_of_actions, dtype=np.float))
    trial_data = [[] for i in range(10)]
    for trials in range(10):
        q_value = defaultdict(lambda: np.zeros(len(env.actions)))
        cum_steps = 0
        for episode in trange(num_episodes):
            state = env.reset()
            episode_steps = []
            episode_complete = True
            for step in range(10000):
                probs = epsilon_greedy_policy(q_value, state, len(env.actions), epsilon)
                action = np.random.choice(np.arange(len(probs)), p=probs)
                next_state, reward, done, _ = env.step(action)
                episode_steps.append((state, action, reward))
                if done:
                    episode_complete = False
                    cum_steps += step
                    trial_data[trials].append(cum_steps)
                    break
                state = next_state
            if episode_complete==True:
                cum_steps += 10000
                trial_data[trials].append(cum_steps)

            G = 0
            for i, (state, action, reward) in enumerate(episode_steps[::-1]):
                G = gamma * G + reward
                previous_states = [(j[0], j[1]) for j in episode_steps[0:len(episode_steps) - i - 1]]
                if (state, action) not in previous_states:
                    count_occurence[state][action] += 1
                    q_value[state][action] = (q_value[state][action] * (count_occurence[state][action] - 1) + G) / (count_occurence[state][action])

    return trial_data

def q4d():
    env = StochasticGrid()
    num_episodes = 250
    trial_data_algorithms = []
    trial_data_mc = monte_carlo_policy_control(env, num_episodes, gamma=1, epsilon=0.1)
    trial_data_q_learning = Q_learning(env, num_episodes, alpha=0.5, gamma=1, epsilon=0.1)
    trial_data_sarsa = SARSA(env, num_episodes, alpha=0.5, gamma=1, epsilon=0.1)
    trial_data_expected_sarsa = expected_SARSA(env, num_episodes, alpha=0.5, gamma=1, epsilon=0.1)
    trial_data_nstep_sarsa = n_step_SARSA(env, num_episodes, n_step=3, alpha=0.5, gamma=1, epsilon=0.1)

    trial_data_algorithms.append(trial_data_nstep_sarsa)
    trial_data_algorithms.append(trial_data_mc)
    trial_data_algorithms.append(trial_data_expected_sarsa)
    trial_data_algorithms.append(trial_data_sarsa)
    trial_data_algorithms.append(trial_data_q_learning)

    algorithms_name = ["n-step SARSA", "monte_carlo", "expected_SARSA", "SARSA", "Q-learning"]
    plot_performance(np.array(trial_data_algorithms), ["black", "orange", "blue", "red", "green"], algorithms_name)
    return

def q4c():
    env = all_direction_Grid()
    num_episodes = 250
    trial_data_algorithms = []
    trial_data_mc = monte_carlo_policy_control(env, num_episodes, gamma=1, epsilon=0.1)
    trial_data_q_learning = Q_learning(env, num_episodes, alpha=0.5, gamma=1, epsilon=0.1)
    trial_data_sarsa = SARSA(env, num_episodes, alpha=0.5, gamma=1, epsilon=0.1)
    trial_data_expected_sarsa = expected_SARSA(env, num_episodes, alpha=0.5, gamma=1, epsilon=0.1)
    trial_data_nstep_sarsa = n_step_SARSA(env, num_episodes, n_step=3, alpha=0.5, gamma=1, epsilon=0.1)

    trial_data_algorithms.append(trial_data_nstep_sarsa)
    trial_data_algorithms.append(trial_data_mc)
    trial_data_algorithms.append(trial_data_expected_sarsa)
    trial_data_algorithms.append(trial_data_sarsa)
    trial_data_algorithms.append(trial_data_q_learning)

    algorithms_name = ["n-step SARSA", "monte_carlo", "expected_SARSA", "SARSA", "Q-learning"]
    plot_performance(np.array(trial_data_algorithms), ["black", "orange", "blue", "red", "green"], algorithms_name)
    return

def q4c2():
    env = extra_move_Grid()
    num_episodes = 250
    trial_data_algorithms = []
    trial_data_mc = monte_carlo_policy_control(env, num_episodes, gamma=1, epsilon=0.1)
    trial_data_q_learning = Q_learning(env, num_episodes, alpha=0.5, gamma=1, epsilon=0.1)
    trial_data_sarsa = SARSA(env, num_episodes, alpha=0.5, gamma=1, epsilon=0.1)
    trial_data_expected_sarsa = expected_SARSA(env, num_episodes, alpha=0.5, gamma=1, epsilon=0.1)
    trial_data_nstep_sarsa = n_step_SARSA(env, num_episodes, n_step=3, alpha=0.5, gamma=1, epsilon=0.1)

    trial_data_algorithms.append(trial_data_nstep_sarsa)
    trial_data_algorithms.append(trial_data_mc)
    trial_data_algorithms.append(trial_data_expected_sarsa)
    trial_data_algorithms.append(trial_data_sarsa)
    trial_data_algorithms.append(trial_data_q_learning)

    algorithms_name = ["n-step SARSA", "monte_carlo", "expected_SARSA", "SARSA", "Q-learning"]
    plot_performance(np.array(trial_data_algorithms), ["black", "orange", "blue", "red", "green"], algorithms_name)

def q4a():
    env = Grid()
    num_episodes = 250
    trial_data_algorithms = []
    trial_data_mc = monte_carlo_policy_control(env, num_episodes, gamma=1, epsilon=0.1)
    trial_data_q_learning = Q_learning(env, num_episodes, alpha = 0.5, gamma=1, epsilon=0.1)
    trial_data_sarsa = SARSA(env, num_episodes, alpha = 0.5, gamma=1, epsilon = 0.1)
    trial_data_expected_sarsa = expected_SARSA(env, num_episodes, alpha=0.5, gamma=1, epsilon=0.1)
    trial_data_nstep_sarsa = n_step_SARSA(env, num_episodes, n_step=3, alpha=0.5, gamma=1, epsilon=0.1)

    trial_data_algorithms.append(trial_data_nstep_sarsa)
    trial_data_algorithms.append(trial_data_mc)
    trial_data_algorithms.append(trial_data_expected_sarsa)
    trial_data_algorithms.append(trial_data_sarsa)
    trial_data_algorithms.append(trial_data_q_learning)

    algorithms_name = ["n-step SARSA", "monte_carlo", "expected_SARSA", "SARSA", "Q-learning"]
    plot_performance(np.array(trial_data_algorithms), ["black", "orange", "blue","red","green"], algorithms_name)
    return

def get_SARSA_policy_control(env, num_episodes=200, alpha=0.5, gamma=1, epsilon=0.1):
    q_value = defaultdict(lambda: np.zeros(len(env.actions)))
    q_policy = defaultdict(lambda: np.array([(1.0 / float(len(env.actions))) for k in np.arange(len(env.actions))]))
    for episode in trange(num_episodes):
        state = env.reset()
        probs = epsilon_greedy_policy(q_value, state, len(env.actions), epsilon)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        for step in range(10000):
            next_state, reward, done, _ = env.step(action)
            probs = epsilon_greedy_policy(q_value, next_state, len(env.actions), epsilon)
            new_action = np.random.choice(np.arange(len(probs)), p=probs)
            td_target = reward + gamma * q_value[next_state][new_action]
            td_error = td_target - q_value[state][action]
            q_value[state][action] += alpha * td_error
            if done:
                break
            state = next_state
            action = new_action
        a_max = np.random.choice(np.where((q_value[state] == np.max(q_value[state])))[0])
        for a in np.arange(len(env.actions)):
            q_policy[state][a] = ((1 - epsilon) + epsilon / float(len(env.actions))) if a == a_max else (epsilon / float(len(env.actions)))
    return q_policy

def evaluation_td(env, training_episodes, testing_episodes, alpha = 0.5, gamma=1, epsilon = 0.1):
    value_function = defaultdict(float)
    value_function[env.goal_state] = 0
    for i in trange(len(training_episodes)):
        for step in range(len(training_episodes[i])-1):
            s,a,r = training_episodes[i][step]
            s_ = training_episodes[i][step+1][0]
            value_function[s] += alpha*(r + gamma*value_function[s_] - value_function[s])

    ##Evaluation
    hist_list = [0 for i in range(100)]
    for e in trange(len(testing_episodes)):
        for step in range(len(testing_episodes[e])-1):
            print(step)
            s,a,r = testing_episodes[e][step]
            s_ = testing_episodes[e][step + 1][0]
            if s==env.start:
                hist_list[e] = r + gamma*value_function[s_]
                break
    return hist_list

def evaluation_n_step_TD(env, training_episodes, testing_episodes, n_step, alpha=0.5, gamma=1, epsilon=0.1):
    value_function = defaultdict(float)
    value_function[env.goal_state] = 0
    for e in trange(len(training_episodes)):
        for step in range(len(training_episodes[e])):
            s, a, r = training_episodes[e][step]
            G = 0
            for i in range(n_step):
                if step+i+1<len(training_episodes[e]):
                    G += training_episodes[e][step + i + 1][2]
            if step+n_step<len(training_episodes):
                G += value_function[training_episodes[e][step + n_step][0]]
            value_function[s] += alpha*(G-value_function[s])
    #Evaluation
    hist_list = [0 for i in range(100)]
    for e in trange(len(testing_episodes)):
        for step in range(len(testing_episodes[e])):
            s, a, r = testing_episodes[e][step]
            if s == env.start:
                G = 0
                for i in range(n_step):
                    if step + i + 1< len(testing_episodes[e]):
                        G += testing_episodes[e][step + i + 1][2]
                if step + n_step < len(testing_episodes):
                    G += value_function[testing_episodes[e][step + n_step][0]]
                hist_list[e] = G
                break

    return hist_list

def evaluation_monte_carlo(env, training_episodes, testing_episodes, gamma=1, epsilon=0.1):
    value_function = defaultdict(float)
    value_function[env.goal_state] = 0
    count_occurence = defaultdict(float)
    for e in trange(len(training_episodes)):
        visited = dict()
        for step in range(len(training_episodes[e])):
            s, a, r = training_episodes[e][step]
            if s not in visited:
                G = 0
                i = step + 1
                while i < len(training_episodes[e]):
                    G += training_episodes[e][i][2]
                    i += 1
                count_occurence[s] += 1
                value_function[s] = (value_function[s] * (count_occurence[s] - 1) + G) / (count_occurence[s])
                visited[s] = True

    hist_list = [0 for i in range(100)]
    for e in trange(len(testing_episodes)):
        for step in range(len(testing_episodes[e])):
            s, a, r = testing_episodes[e][step]
            if s == env.start:
                G = 0
                i = step+1
                while i<len(testing_episodes[e]):
                    G += testing_episodes[e][i][2]
                    i += 1
                hist_list[e] = G
                break

    return hist_list

def generate_episodes(env, q_value, num, epsilon=0.1):
    episode_set = []
    state = env.reset()
    for e in trange(num):
        each_episode_set = []
        for s in range(10000):
            probs = epsilon_greedy_policy(q_value, state, len(env.actions), epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            each_episode_set.append((state, action, reward))
            if done:
                break
            state = next_state
        episode_set.append(each_episode_set)
    return episode_set



def q5a():
    env = Grid()
    q_policy = get_SARSA_policy_control(env, num_episodes=200, alpha=0.5, gamma=1, epsilon=0.1)
    episodes = [1,10,50]
    testing_episodes = generate_episodes(env, q_policy, 100, 0.1)
    for epi in episodes:
        training_episodes = generate_episodes(env, q_policy, epi, 0.1)
        # hist_list = evaluation_td(env, training_episodes, testing_episodes, alpha = 0.5, gamma=1, epsilon = 0.1)
        # plot_histogram(hist_list, "red", "TD evaluation")
        hist_list = evaluation_n_step_TD(env, training_episodes, testing_episodes, 4, alpha=0.5, gamma=1, epsilon=0.1)
        plot_histogram(hist_list, "red", "n-step TD")
        # hist_list = evaluation_monte_carlo(env, training_episodes, testing_episodes, gamma=1, epsilon=0.1)
        # plot_histogram(hist_list, "red", "monte-carlo")



if __name__ == "__main__":
    # q4a()
    # q4c2()
    # q4c()
    q4d()
    # q5a()
