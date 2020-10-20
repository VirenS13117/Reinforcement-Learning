import gym
import numpy as np
from Ex4.src.Policy import Policy

env = gym.make("Blackjack-v0")

def get_average_return(returns):
    return np.average(returns)

def first_visit_monte_carlo_policy_evaluation(policy, gamma=0.1):
    value_function = [0 for i in range(32)]
    returns = dict()
    for i in range(32):
        returns[i] = []
    num_episodes = 100
    episode = []
    for i in range(num_episodes):
        env.reset()
        states = []
        actions = []
        rewards = [0]
        state = env._get_obs()[0]
        isTerminal = False
        t = 0
        while not isTerminal:
            states.append(state)
            action = policy.get_action(state)
            actions.append(action)
            t += 1
            state, reward, isTerminal, info = env.step(action)
            rewards.append(reward)
            state = state[0]
        t -= 1
        G = 0
        while t>0:
            G = gamma*G + rewards[t+1]
            print("G : ", G)
            print("current state : ", states[t], "old states : ",states[0:t])
            if states[t] not in states[0:t]:
                returns[states[t]].append(G)
                value_function[states[t]] = get_average_return(returns[states[t]])
            t -= 1

        print(G)
        print("states : ", states)
        print("actions : ", actions)
        print("rewards : ", rewards)
        print("returns : ", returns)
        print("value function : ", value_function)




if __name__ == "__main__":
    print(env.observation_space[0])
    policy = Policy("sticks on 20-21")
    first_visit_monte_carlo_policy_evaluation(policy, 0.1)
    print(len(env.observation_space))
