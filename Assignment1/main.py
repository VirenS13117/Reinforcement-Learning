# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# generate random integer values
from Assignment1 import environment, Simulation, randomPolicy, worsePolicy, betterPolicy, Simulation

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Create ")
    source = (0,0)
    target = (10,10)
    my_grid = environment.Environment(source, target)
    worse_policy = worsePolicy.WorsePolicy()
    random_policy = randomPolicy.RandomPolicy()
    better_policy = betterPolicy.BetterPolicy()
    simulation_object_random = Simulation.Simulation(my_grid, random_policy)
    simulation_object_random.simulate()
    # simulation_object_worse = Simulation.Simulation(my_grid, worse_policy)
    # simulation_object_worse.simulate()
    # simulation_object_better = Simulation.Simulation(my_grid, better_policy)
    # simulation_object_better.simulate()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
