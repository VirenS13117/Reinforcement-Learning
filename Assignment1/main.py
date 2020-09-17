# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# generate random integer values
from Assignment1 import environment, q1, plot_rewards, randomPolicy, worsePolicy
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Create ")
    # q1.play_simulation()
    worse_policy = worsePolicy.GoWest()
    random_policy = randomPolicy.RandomPolicy()
    plot_rewards.cumulative_reward(random_policy)
    plot_rewards.cumulative_reward(worse_policy)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
