from Ex3.src.Action import Action
from Ex3.src.Transition import Transition
from Ex3.src.GridWorld import GridWorld
from Ex3.src.Policy import Policy
import numpy as np


def iterative_policy_evaluation(gw, policy, value_state, gamma, theta=10 ** -3):
    while True:
        delta = 0
        for row in range(gw.rows):
            for col in range(gw.cols):
                curr_state = gw.get_state_id((row, col))
                new_state_value = 0
                for action in gw.get_actions():
                    next_state, reward, prob = gw.transitions.next(curr_state, action)
                    newx, newy = gw.get_state_position(next_state)
                    new_state_value += policy.get_probability(curr_state, action) * prob * (reward + gamma * value_state[newx][newy])
                delta = max(delta, abs(value_state[row][col] - new_state_value))
                value_state[row][col] = new_state_value
        if delta < theta:
            break
        print(delta)
    return np.round(value_state, 1)


def value_iteration(gw, policy, value_state, gamma, theta=10 ** -3):
    while True:
        delta = 0
        for row in range(gw.rows):
            for col in range(gw.cols):
                curr_state = gw.get_state_id((row, col))
                new_state_value = 0
                for action in gw.get_actions():
                    next_state, reward, prob = gw.transitions.next(curr_state, action)
                    newx, newy = gw.get_state_position(next_state)
                    new_state_value = max(new_state_value, prob*(reward + gamma*value_state[newx][newy]))
                delta = max(delta, abs(value_state[row][col] - new_state_value))
                value_state[row][col] = new_state_value
        if delta < theta:
            break
    value_state = np.round(value_state, 1)
    for row in range(gw.rows):
        for col in range(gw.cols):
            curr_state = gw.get_state_id((row,col))
            val = np.array([])
            for action in gw.get_actions():
                next_state, reward, prob = gw.transitions.next(curr_state, action)
                newx, newy = gw.get_state_position(next_state)
                val = np.append(val, (prob * (reward + gamma*value_state[newx][newy])))

            max_act = np.where(val==np.max(val))[0]
            if len(max_act):
                for i, action in enumerate(gw.get_actions()):
                    if i in max_act:
                        policy.policy_map(curr_state, action, 1/len(max_act))
                    else:
                        policy.policy_map(curr_state, action, 0)
    return value_state, policy


def policy_iteration(gw, policy, value_state, gamma, theta=10 ** -3):
    policy_stable = False
    while not policy_stable:
        policy_stable = True
        value_state = iterative_policy_evaluation(gw, policy, value_state, gamma, theta)
        for row in range(gw.rows):
            for col in range(gw.cols):
                curr_state = gw.get_state_id((row, col))
                old_best_action = np.array([])
                val = np.array([])
                for i, action in enumerate(gw.get_actions()):
                    next_state, reward, prob = gw.transitions.next(curr_state, action)
                    newx, newy = gw.get_state_position(next_state)
                    val = np.append(val, (prob * (reward + gamma * value_state[newx][newy])))
                    if policy.get_probability(curr_state, action):
                        old_best_action = np.append(old_best_action, i)

                max_act = np.where(val == np.max(val))[0]
                if len(max_act) > 0:
                    for i, action in enumerate(gw.get_actions()):
                        if i in max_act:
                            policy.policy_map(curr_state, action, 1 / len(max_act))
                        else:
                            policy.policy_map(curr_state, action, 0)
                if policy_stable and (np.array_equal(old_best_action, max_act)==False):
                    policy_stable = False
    return value_state, policy



def q_a():
    actions = {"north": (-1,0), "south": (1,0), "east": (0,1), "west": (0,-1)}
    gw = GridWorld(5, 5, actions, dr = 0, oob_r = -1, sr={1:(21,10), 3:(13,5)})
    vs = np.zeros((gw.rows, gw.cols))
    pi = Policy(gw.get_states(), gw.get_actions())
    state_values = iterative_policy_evaluation(gw, pi, vs, gamma=0.9)
    print(np.array(state_values))
    return

def q_b():
    actions = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
    gw = GridWorld(5, 5, actions, dr=0, oob_r=-1, sr={1: (21, 10), 3: (13, 5)})
    vs = np.zeros((gw.rows, gw.cols))
    pi = Policy(gw.get_states(), gw.get_actions())
    pi.plot(gw)
    state_value, new_pi = value_iteration(gw,pi,vs,gamma=0.9)
    print(np.array(state_value))
    new_pi.plot(gw)
    return

def q_c():
    actions = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
    gw = GridWorld(5, 5, actions, dr=0, oob_r=-1, sr={1: (21, 10), 3: (13, 5)})
    vs = np.zeros((gw.rows, gw.cols))
    pi = Policy(gw.get_states(), gw.get_actions())
    pi.plot(gw)
    state_value, new_pi = policy_iteration(gw, pi, vs, gamma=0.9)
    print(np.array(state_value))
    new_pi.plot(gw)
    return

if __name__ == "__main__":
    # q_a()
    # q_b()
    q_c()