import matplotlib
import numpy as np
import gym
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from mpl_toolkits.mplot3d import Axes3D
import sklearn.pipeline
from tqdm import trange
import sklearn.preprocessing
from Ex7.src.tiles3 import IHT, tiles



env = gym.make('MountainCar-v0')

discount_factor = 1
nA = env.action_space.n
numTilings = 8
maxSize = 4096
iht = IHT(maxSize)

plt_actions = np.zeros(nA)


def featurize_state(state, action=0):
    x, y = state
    featurized = tiles(iht, 8, [8 * x / (0.5 + 1.2), 8 * y / (0.07 + 0.07)], [action])
    features = np.zeros((8,maxSize))
    features = [0 for i in range(maxSize)]
    for i in featurized:
        features[i] = 1
    return features


def Q(state, action, w):
    state_features = featurize_state(state, action)
    value = np.dot(state_features,w)
    return value


def policy(state, w, epsilon=0):
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax([Q(state, a, w) for a in range(nA)])
    A[best_action] += (1.0 - epsilon)
    sample = np.random.choice(nA, p=A)
    return sample


def check_gradients(index, state, next_state, target, action, next_action, weight, reward):
    ew1 = np.array(weight, copy=True)
    ew2 = np.array(weight, copy=True)
    epsilon = 1e-6
    ew1[action][index] += epsilon
    ew2[action][index] -= epsilon

    test_target_1 = reward + discount_factor * Q(next_state, next_action, ew1)
    td_error_1 = target - Q(state, action, ew1)

    test_target_2 = reward + discount_factor * Q(next_state, next_action, ew2)
    td_error_2 = target - Q(state, action, ew2)

    grad = (td_error_1 - td_error_2) / (2 * epsilon)

    return grad[0]

def one_step_SARSA(alpha, w):
    state = env.reset()
    step_count = 0
    while True:
        step_count += 1
        action = policy(state, w)
        plt_actions[action] += 1
        next_state, reward, done, _ = env.step(action)
        if done:
            td_error = Q(state, action, w) - reward
            dw = np.dot(td_error, featurize_state(state, action))
            w -= alpha * dw
            break
        next_action = policy(next_state,w)
        target = reward + discount_factor * Q(next_state, next_action, w)
        td_error = Q(state, action, w) - target
        dw = np.dot(td_error, featurize_state(state, action))
        w -= alpha*dw
        state = next_state
    return w, step_count


def plot_cost_to_go_mountain_car(w, num_episodes, num_tiles=64):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max([Q(_, a, w) for a in range(nA)]), 2, np.dstack([X, Y]))
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function, Total Episodes = " +  str(num_episodes))
    fig.colorbar(surf)
    plt.show()


def avg_steps_per_episode():
    alphas = [0.1/8, 0.2/8, 0.5/8]
    alpha_names = ["0.1/8", "0.2/8", "0.5/8"]
    num_episode = 500
    num_avg = 10
    steps_per_episode = np.zeros((len(alphas), num_episode))
    for k in range(num_avg):
        for i in range(len(alphas)):
            alpha = alphas[i]
            w = np.zeros(maxSize)
            for j in trange(num_episode):
                w, step = one_step_SARSA(alpha, w)
                steps_per_episode[i,j] += step
    for i in range(len(alphas)):
        for j in range(num_episode):
            steps_per_episode[i,j] /= num_avg

    e = [i for i in range(num_episode)]
    for i in range(0, len(alphas)):
        plt.plot(e, steps_per_episode[i], label='alpha = '+alpha_names[i])
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode')
    plt.yscale('log')
    plt.legend()
    plt.show()

if __name__=="__main__":
    number_of_episodes = [1,12,104,1000,9000]
    for i in range(len(number_of_episodes)):
        w = np.zeros(maxSize)
        episodes = number_of_episodes[i]
        for j in trange(episodes):
            w, step = one_step_SARSA(0.1,w)
        plot_cost_to_go_mountain_car(w, episodes)
    # avg_steps_per_episode()
    env.close()