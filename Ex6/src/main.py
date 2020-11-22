import numpy as np
from typing import Tuple, List, Set, Dict, Callable
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from tqdm import trange
from Ex6.src.Environment import Grid
from Ex6.src.Model import Model
from Ex6.src.Stochastic_Environment import StochasticGrid
from Ex6.src.Stochastic_Model import Stochastic_Model

n = 250
alpha = 0.1
epsilon = 0.1
gamma = 0.95
kappa = 0.01

def epsilon_greedy_policy(Q, state, nA, epsilon):
    probs = np.ones(nA) * epsilon / nA
    best_action = np.argmax(Q[state])
    probs[best_action] += 1.0 - epsilon
    return probs


def plot_performance(rewards: np.ndarray, labels: List[str], clrs: List[str], title: str):
    fig, s_plt = plt.subplots(1, figsize=(8, 6), facecolor="white")
    xdata = [x for x in range(rewards.shape[2])]
    for i in range(rewards.shape[0]):
        ydata = np.average(rewards[i], axis=0)
        s_plt.plot(xdata, ydata, color=clrs[i], label=labels[i])
        ystderr = np.std(rewards[i], axis=0)
        ystderr *= 1 / math.sqrt(rewards.shape[1])
        ystderr *= 1.96
        s_plt.fill_between(xdata, np.subtract(ydata, ystderr), np.add(ydata, ystderr), alpha=0.2, color=clrs[i])
    plt.legend(loc="upper left")
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Rewards")
    plt.title(title + ":" + " Time Steps = " + str(rewards.shape[2]) + "," + " Trials = " + str(rewards.shape[1]))
    plt.show()
    return


def dynaQ_footnote_v1(blocklist1, blocklist2, n_steps, change_time):
    env = Grid(blocklist1)
    n_actions = len(env.actions)
    n_states = (env.max_x - env.min_x + 1) * (env.max_y - env.min_y + 1)
    print(n_states)
    q_value = defaultdict(lambda: np.zeros(len(env.actions)))
    model = Model(env, n_states, n_actions)
    state = env.reset()
    cum_reward = [0]
    for time_step in trange(1, n_steps + 1):
        if np.random.uniform() < epsilon:
            action = np.random.choice([i for i in range(len(env.actions))])
        else:
            action = np.random.choice(np.where(q_value[state] == np.max(q_value[state]))[0])
        next_state, reward, done, _ = env.step(action)
        td_target = reward + gamma * np.amax(q_value[next_state])
        td_error = td_target - q_value[state][action]
        q_value[state][action] += alpha * td_error
        model.add(state, action, next_state, reward)
        for i in range(n):
            s, a = model.sample()
            a = np.random.choice([i for i in range(len(env.actions))])
            s_prime, r, t = model.step(s, a)
            q_value[s][a] += alpha * (r + gamma * np.amax(q_value[s_prime]) - q_value[s][a])
        state = next_state
        if done:
            state = env.reset()
        if time_step == change_time:
            env.change_blocklist(blocklist2)
            state = env.reset()
            for key in list(model.transitions):
                if model.transitions[key] in env.blocks or key[0] in env.blocks:
                    model.transitions.pop(key, None)
                    q_value.pop(key, None)
        cum_reward.append(cum_reward[-1] + reward)
    return np.array(cum_reward[1:])

def dynaQ_footnote_v2(blocklist1, blocklist2, n_steps, change_time):
    env = Grid(blocklist1)
    n_actions = len(env.actions)
    n_states = (env.max_x - env.min_x + 1) * (env.max_y - env.min_y + 1)
    print(n_states)
    q_value = defaultdict(lambda: np.zeros(len(env.actions)))
    model = Model(env, n_states, n_actions)
    for i in range(n_states):
        for j in range(n_actions):
            model.transitions[(i, j)] = (i, j)
            model.rewards[(i, j)] = 0
            model.time[(i,j)] = 0
    state = env.reset()
    cum_reward = [0]
    for time_step in trange(1, n_steps + 1):
        if np.random.uniform() < epsilon:
            action = np.random.choice([i for i in range(len(env.actions))])
        else:
            action = np.random.choice(np.where(q_value[state] == np.max(q_value[state]))[0])
        next_state, reward, done, _ = env.step(action)
        td_target = reward + gamma * np.amax(q_value[next_state])
        td_error = td_target - q_value[state][action]
        q_value[state][action] += alpha * td_error
        model.add(state, action, next_state, reward)
        for i in range(n):
            s, a = model.sample()
            a = np.random.choice([i for i in range(len(env.actions))])
            s_prime, r, t = model.step(s, a)
            q_value[s][a] += alpha * (r + gamma * np.amax(q_value[s_prime]) - q_value[s][a])
        state = next_state
        if done:
            state = env.reset()
        if time_step == change_time:
            env.change_blocklist(blocklist2)
            state = env.reset()
            for key in list(model.transitions):
                if model.transitions[key] in env.blocks or key[0] in env.blocks:
                    model.transitions[key] = key
                    q_value.pop(key, None)
        cum_reward.append(cum_reward[-1] + reward)
    return np.array(cum_reward[1:])

def dynaQ(blocklist1, blocklist2, n_steps, change_time):
    env = Grid(blocklist1)
    n_actions = len(env.actions)
    n_states = (env.max_x - env.min_x + 1) * (env.max_y - env.min_y + 1)
    print(n_states)
    q_value = defaultdict(lambda: np.zeros(len(env.actions)))
    model = Model(env, n_states, n_actions)
    state = env.reset()
    cum_reward = [0]
    for time_step in trange(1, n_steps + 1):
        if np.random.uniform() < epsilon:
            action = np.random.choice([i for i in range(len(env.actions))])
        else:
            action = np.random.choice(np.where(q_value[state] == np.max(q_value[state]))[0])
        next_state, reward, done, _ = env.step(action)
        td_target = reward + gamma * np.amax(q_value[next_state])
        td_error = td_target - q_value[state][action]
        q_value[state][action] += alpha * td_error
        model.add(state, action, next_state, reward)
        for i in range(n):
            s, a = model.sample()
            s_prime, r, t = model.step(s, a)
            q_value[s][a] += alpha * (r + gamma * np.amax(q_value[s_prime]) - q_value[s][a])
        state = next_state
        if done:
            state = env.reset()
        if time_step == change_time:
            env.change_blocklist(blocklist2)
            state = env.reset()
            for key in list(model.transitions):
                if model.transitions[key] in env.blocks or key[0] in env.blocks:
                    model.transitions.pop(key, None)
                    q_value.pop(key, None)
        cum_reward.append(cum_reward[-1] + reward)
    return np.array(cum_reward[1:])

def dynaQ_plus_updated_action_selection(blocklist1, blocklist2, n_steps, change_time):
    env = Grid(blocklist1)
    n_actions = len(env.actions)
    n_states = (env.max_x - env.min_x + 1) * (env.max_y - env.min_y + 1)
    q_value = defaultdict(lambda: np.zeros(len(env.actions)))
    model = Model(env, n_states, n_actions)
    state = env.reset()
    cum_reward = [0]
    for time_step in trange(1, n_steps + 1):
        if np.random.uniform() < epsilon:
            action = np.random.choice([i for i in range(len(env.actions))])
        else:
            action_values = []
            for a in range(n_actions):
                tau = time_step - model.time.get((state,a),0)
                action_values.append(q_value[state][a] + kappa*np.sqrt(tau))
            action = np.argmax(action_values)
        next_state, reward, done, _ = env.step(action)
        td_target = reward + gamma * np.amax(q_value[next_state])
        td_error = td_target - q_value[state][action]
        q_value[state][action] += alpha * td_error
        model.add(state, action, next_state, reward, time_step)
        for i in range(n):
            s, a = model.sample()
            s_prime, r, t = model.step(s, a)
            q_value[s][a] += alpha * (r + gamma * np.amax(q_value[s_prime]) - q_value[s][a])
        state = next_state
        if done:
            state = env.reset()
        if time_step == change_time:
            env.change_blocklist(blocklist2)
            state = env.reset()
            for key in list(model.transitions):
                if model.transitions[key] in env.blocks or key[0] in env.blocks:
                    model.transitions.pop(key, None)
                    q_value.pop(key, None)
        cum_reward.append(cum_reward[-1] + reward)
    return np.array(cum_reward[1:])


def dynaQ_plus(blocklist1, blocklist2, n_steps, change_time):
    env = Grid(blocklist1)
    n_actions = len(env.actions)
    n_states = (env.max_x - env.min_x + 1) * (env.max_y - env.min_y + 1)
    q_value = defaultdict(lambda: np.zeros(len(env.actions)))
    model = Model(env, n_states, n_actions)
    state = env.reset()
    cum_reward = [0]
    for time_step in trange(1, n_steps + 1):
        if np.random.uniform() < epsilon:
            action = np.random.choice([i for i in range(len(env.actions))])
        else:
            action = np.random.choice(np.where(q_value[state] == np.max(q_value[state]))[0])
        next_state, reward, done, _ = env.step(action)
        td_target = reward + gamma * np.amax(q_value[next_state])
        td_error = td_target - q_value[state][action]
        q_value[state][action] += alpha * td_error
        model.add(state, action, next_state, reward, time_step)
        for i in range(n):
            s, a = model.sample()
            s_prime, r, t = model.step(s, a)
            tau = time_step - t
            r = r + kappa * np.sqrt(tau)
            model.time[(s,a)] = time_step
            q_value[s][a] += alpha * (r + gamma * np.amax(q_value[s_prime]) - q_value[s][a])
        state = next_state
        if done:
            state = env.reset()
        if time_step == change_time:
            env.change_blocklist(blocklist2)
            state = env.reset()
            for key in list(model.transitions):
                if model.transitions[key] in env.blocks or key[0] in env.blocks:
                    model.transitions.pop(key, None)
                    q_value.pop(key, None)
        cum_reward.append(cum_reward[-1] + reward)
    return np.array(cum_reward[1:])

def dynaQ_stochastic_consistent_environment(blocklist1, blocklist2, n_steps, change_time):
    env = StochasticGrid(blocklist1)
    n_actions = len(env.actions)
    n_states = (env.max_x - env.min_x + 1) * (env.max_y - env.min_y + 1)
    print(n_states)
    q_value = defaultdict(lambda: np.zeros(len(env.actions)))
    model = Stochastic_Model(env)
    state = env.reset()
    cum_reward = [0]
    for time_step in trange(1, n_steps + 1):
        if np.random.uniform() < epsilon:
            action = np.random.choice([i for i in range(len(env.actions))])
        else:
            action = np.random.choice(np.where(q_value[state] == np.max(q_value[state]))[0])
        next_state, reward, done, state_prob_list = env.step(action)
        td_target = reward + gamma * np.amax(q_value[next_state])
        td_error = td_target - q_value[state][action]
        q_value[state][action] += alpha * td_error
        model.add(state, action, state_prob_list)
        for i in range(n):
            s, a = model.sample()
            s_prime, r = model.step(s, a)
            q_value[s][a] += alpha * (r + gamma * np.amax(q_value[s_prime]) - q_value[s][a])
        state = next_state
        if done:
            state = env.reset()
        cum_reward.append(cum_reward[-1] + reward)
    return np.array(cum_reward[1:])

def dynaQ_stochastic_changing_environment(blocklist1, blocklist2, n_steps, change_time):
    env = StochasticGrid(blocklist1)
    n_actions = len(env.actions)
    n_states = (env.max_x - env.min_x + 1) * (env.max_y - env.min_y + 1)
    print(n_states)
    q_value = defaultdict(lambda: np.zeros(len(env.actions)))
    model = Stochastic_Model(env)
    state = env.reset()
    cum_reward = [0]
    for time_step in trange(1, n_steps + 1):
        if np.random.uniform() < epsilon:
            action = np.random.choice([i for i in range(len(env.actions))])
        else:
            action = np.random.choice(np.where(q_value[state] == np.max(q_value[state]))[0])
        next_state, reward, done, state_prob_list = env.step(action)
        td_target = reward + gamma * np.amax(q_value[next_state])
        td_error = td_target - q_value[state][action]
        q_value[state][action] += alpha * td_error
        model.add(state, action, state_prob_list)
        for i in range(n):
            s, a = model.sample()
            s_prime, r = model.step(s, a)
            q_value[s][a] += alpha * (r + gamma * np.amax(q_value[s_prime]) - q_value[s][a])
        state = next_state
        if done:
            state = env.reset()
        if time_step == change_time:
            env.change_blocklist(blocklist2)
            state = env.reset()
            for key in list(model.transitions):
                if model.transitions[key] in env.blocks or key[0] in env.blocks:
                    model.transitions.pop(key, None)
                    q_value.pop(key, None)
        cum_reward.append(cum_reward[-1] + reward)
    return np.array(cum_reward[1:])

def plot_data(data, name):
    x = np.arange(len(data))
    plt.plot(x, data, '-', markersize=2, label=name)
    plt.legend(loc='lower right', prop={'size': 16}, numpoints=5)
    plt.show()

def q4(blocklist1, blocklist2, n_steps = 6000, change_time = 3000):
    cum_rewards_dynaQ = []
    cum_rewards_dynaQ_plus_updated_action_selection = []
    cum_rewards_dynaQ_plus = []
    for i in range(10):
        cum_rewards_dynaQ.append(dynaQ(blocklist1, blocklist2, n_steps, change_time))
        cum_rewards_dynaQ_plus.append(dynaQ_plus(blocklist1, blocklist2, n_steps, change_time))
        cum_rewards_dynaQ_plus_updated_action_selection.append(dynaQ_plus_updated_action_selection(blocklist1, blocklist2, n_steps, change_time))
    dynaQ_rewards = [cum_rewards_dynaQ, cum_rewards_dynaQ_plus, cum_rewards_dynaQ_plus_updated_action_selection]
    plot_performance(np.array(dynaQ_rewards), ["DynaQ_plus", "DynaQ", "DynaQ_plus_updated_action_selection"], ["blue", "red", "green"], "Q4 : DynaQ, DynaQ+, DynaQ+(Updated action selection)")
    return

def q5(blocklist1, blocklist2, n_steps = 6000, change_time = 3000):
    cum_rewards_dynaQ_consistent = []
    cum_rewards_dynaQ_changing  = []
    for i in range(10):
        cum_rewards_dynaQ_consistent.append(dynaQ_stochastic_consistent_environment(blocklist1, blocklist2, n_steps, change_time))
        cum_rewards_dynaQ_changing.append(dynaQ_stochastic_changing_environment(blocklist1, blocklist2, n_steps, change_time))
    dynaQ_rewards = [cum_rewards_dynaQ_consistent, cum_rewards_dynaQ_changing]
    plot_performance(np.array(dynaQ_rewards), ["DynaQ_stochastic_environment", "DynaQ_stochastic_changing_environment"], ["blue","red"], "stochastic and changing environments")
    return

def q3(blocklist1, blocklist2, n_steps = 6000, change_time = 3000):
    cum_rewards_dynaQ = []
    cum_rewards_dynaQ_plus = []
    cum_rewards_dynaQ_plus_footnote_v1 = []
    cum_rewards_dynaQ_plus_footnote_v2 = []
    for i in range(10):
        cum_rewards_dynaQ.append(dynaQ(blocklist1, blocklist2, n_steps, change_time))
        cum_rewards_dynaQ_plus.append(dynaQ_plus(blocklist1, blocklist2, n_steps, change_time))
        # cum_rewards_dynaQ_plus_footnote_v1.append(dynaQ_footnote_v1(blocklist1, blocklist2, n_steps, change_time))
        # cum_rewards_dynaQ_plus_footnote_v2.append(dynaQ_footnote_v2(blocklist1, blocklist2, n_steps, change_time))
    # dynaQ_rewards = [cum_rewards_dynaQ, cum_rewards_dynaQ_plus_footnote_v1, cum_rewards_dynaQ_plus_footnote_v2]
    dynaQ_rewards = [cum_rewards_dynaQ, cum_rewards_dynaQ_plus]
    # plot_performance(np.array(dynaQ_rewards), ["DynaQ_plus","DynaQ_plus_footnote1", "DynaQ_plus_footnote2"], ["blue","green","red"], "cumulative reward with time steps")
    plot_performance(np.array(dynaQ_rewards), ["DynaQ_plus", "DynaQ"], ["red", "green"],"cumulative reward with time steps")
    return


if __name__ == "__main__":
    blocklist1 = [(i, 2) for i in range(8)]
    blocklist2 = [(i, 2) for i in range(1, 9)]
    blocklist3 = [(i, 2) for i in range(1, 8)]
    q3(blocklist1, blocklist2, n_steps = 3000, change_time = 1000)
    # q3(blocklist2, blocklist3, n_steps = 6000, change_time = 3000)
    # q5(blocklist1, blocklist2, n_steps = 3000, change_time = 1000)
    # q5(blocklist2, blocklist3, n_steps = 6000, change_time = 3000)
    # q4(blocklist1, blocklist2, n_steps=3000, change_time=1000)
    # q4(blocklist2, blocklist3, n_steps = 6000, change_time = 3000)

