from typing import Tuple, List, Dict, Set
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
from scipy.stats import poisson

car_return_prob = dict()


def poisson_prob(x, h):
    if (x, h) not in car_return_prob:
        car_return_prob[(x, h)] = poisson.pmf(x, h)
    return car_return_prob[(x, h)]


def state_value(state, action, value_state, param_dict, const_return=True):
    expected_return = param_dict["move_car_cost"] * (abs(action)-1)
    first_cars_number = min(state[0] - action, param_dict["max_cars"])
    second_cars_number = min(state[1] + action, param_dict["max_cars"])
    for first_rental in range(first_cars_number):
        for second_rental in range(second_cars_number):
            rental_probability = poisson_prob(first_rental, param_dict["first_average_rental"]) * poisson_prob(
                second_rental, param_dict["second_average_rental"])
            first_available_cars = min(first_cars_number, first_rental)
            second_available_cars = min(second_cars_number, second_rental)
            reward = (first_available_cars + second_available_cars) * param_dict["rental_credit"]
            first_remaining_car = first_cars_number - first_available_cars
            second_remaining_car = second_cars_number - second_available_cars
            if const_return:
                first_remaining_car = min(first_remaining_car + param_dict["first_average_return"],
                                          param_dict["max_cars"])
                second_remaining_car = min(second_remaining_car + param_dict["second_average_return"],
                                           param_dict["max_cars"])
                if first_remaining_car>10:
                    expected_return -= 4
                if second_remaining_car>10:
                    expected_return -= 4
                expected_return += rental_probability * (
                            reward + param_dict["discount"] * value_state[first_remaining_car][second_remaining_car])
            else:
                for first_return in range(param_dict["max_itr"]):
                    for second_return in range(param_dict["max_itr"]):
                        rental_probability = poisson_prob(first_return,
                                                          param_dict["first_average_rental"]) * poisson_prob(
                            second_return, param_dict["second_average_rental"])
                        first_remaining_car = min(first_remaining_car + param_dict["first_average_return"],
                                                  param_dict["max_cars"])
                        second_remaining_car = min(second_remaining_car + param_dict["second_average_return"],
                                                   param_dict["max_cars"])
                        expected_return += rental_probability * (
                                    reward + param_dict["discount"] * value_state[first_remaining_car][
                                second_remaining_car])
    return expected_return


def policy_iteration(states, actions, param_dict, theta=10**-4):
    value_table = np.zeros((param_dict["max_cars"] + 1, param_dict["max_cars"] + 1))
    policy_table = np.zeros(value_table.shape, dtype=np.int)
    optimal_policy = []
    policy_stable = False
    while not policy_stable:
        policy_stable = True
        optimal_policy.append(policy_table.copy())
        while True:
            delta = 0
            old_value_table = value_table.copy()
            for i in states:
                for j in states:
                    value_table[i][j] = state_value((i, j), policy_table[i][j], value_table, param_dict)
            delta = max(delta, abs(old_value_table - value_table).max())
            # print(delta)
            if delta < theta:
                break
        for i in states:
            for j in states:
                action_values = []
                for action in actions:
                    if (0 <= action <= i) or (-j <= action <= 0):
                        action_values.append(state_value((i, j), action, value_table, param_dict))
                    else:
                        action_values.append(-np.inf)
                old_act = policy_table[i][j]
                policy_table[i][j] = actions[np.random.choice(np.where(action_values == np.max(action_values))[0])]
                if policy_stable and old_act != policy_table[i][j]:
                    policy_stable = False
    return np.array(optimal_policy), value_table


def plot_image(nrows, ncols, plt_index, plt, xlabel, ylabel, title, heatmap, param_dict):
    ax = plt.subplot(nrows, ncols, plt_index)
    ax.set_xticks([x for x in range(0, param_dict["max_itr"], 2)])
    ax.set_yticks([y for y in range(0, param_dict["max_itr"], 2)])
    plt.imshow(heatmap)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.subplots_adjust(hspace=0.3)
    next_plot_index = plt_index + 1
    return next_plot_index


if __name__ == "__main__":
    param_dict = dict()
    param_dict = {"max_cars": 20, "max_move": 5,
                  "move_car_cost": 2,
                  "first_average_rental": 3,
                  "second_average_rental": 4,
                  "first_average_return": 3,
                  "second_average_return": 2,
                  "rental_credit": 10,
                  "discount": 0.9,
                  "max_itr": 21}
    print(param_dict["rental_credit"])
    states = [s for s in range(param_dict["max_cars"] + 1)]
    actions = np.arange(-param_dict["max_move"], param_dict["max_move"] + 1)
    optimal_policy, value_table = policy_iteration(states, actions, param_dict)

    plt.figure(figsize=(25, 12.5), facecolor='white')
    plt_index = 1
    r = int(optimal_policy.shape[0] / 2)
    c = int(optimal_policy.shape[0] / 2) + (optimal_policy.shape[0] % 2)
    for i in range(optimal_policy.shape[0]):
        plt_index = plot_image(r, c, plt_index, plt, "cars at position 2", "cars at position 1", "$\pi_{" + str(i) + "}$",
                               np.flipud(optimal_policy[i]), param_dict)
    ax = plt.subplot(r, c, plt_index, projection='3d')
    ax.plot_wireframe([x for x in range(param_dict["max_itr"])], [y for y in range(param_dict["max_itr"])],
                      np.flipud(value_table), color='orangered')

    # Plot show
    plt.show()
