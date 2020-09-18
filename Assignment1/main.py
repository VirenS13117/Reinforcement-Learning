# This is a sample Python script.
from random import randint
import random
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# generate random integer values
from Assignment1 import environment, State, Action, randomPolicy, worsePolicy, betterPolicy, Simulation, manualPolicy, learnedPolicy

random.seed()


def q1():
    source = (0, 0)
    target = (10, 10)
    my_grid = environment.Environment(source, target)
    print("Enter current state : ")
    x = int(input())
    y = int(input())
    curr_state = State.State(x, y, my_grid)
    print("Enter desired action : ")
    action = input()
    action_object = Action.Action(action)
    random_policy = randomPolicy.RandomPolicy(my_grid)
    simulation_object_random = Simulation.Simulation(my_grid, random_policy)
    curr_state, reward = simulation_object_random.transition(curr_state, action_object)
    print("Next state : ", curr_state.x, " ", curr_state.y)
    print("Reward obtained : ", reward)
    return


def q2():
    source = (0, 0)
    target = (10, 10)
    my_grid = environment.Environment(source, target)
    curr_state = State.State(source[0], source[1], my_grid)
    manual_policy = manualPolicy.ManualPolicy()
    simulation_object_manual = Simulation.Simulation(my_grid, manual_policy)
    while True:
        action = manual_policy.get_action(curr_state)
        action_object = Action.Action(action)
        curr_state, reward = simulation_object_manual.transition(curr_state, action_object)
        print("Next state : ", curr_state.x, " ", curr_state.y)
        print("Reward obtained : ", reward)
    return


def q3():
    source = (0, 0)
    target = (10, 10)
    my_grid = environment.Environment(source, target)
    random_policy = randomPolicy.RandomPolicy(my_grid)
    simulation_object_random = Simulation.Simulation(my_grid, random_policy)
    simulation_object_random.simulate()
    return


def q4():
    source = (0, 0)
    # fixed target
    target = (10, 10)
    my_grid = environment.Environment(source, target)
    worse_policy = worsePolicy.WorsePolicy(my_grid)
    random_policy = randomPolicy.RandomPolicy(my_grid)
    better_policy = betterPolicy.BetterPolicy(my_grid)
    simulation_object_random = Simulation.Simulation(my_grid, random_policy)
    simulation_object_random.simulate()
    simulation_object_worse = Simulation.Simulation(my_grid, worse_policy)
    simulation_object_worse.simulate()
    simulation_object_better = Simulation.Simulation(my_grid, better_policy)
    simulation_object_better.simulate()
    return


def q5():
    source = (0, 0)
    # random target
    target = (randint(0, 10), randint(0, 10))
    my_grid = environment.Environment(source, target)
    random_policy = randomPolicy.RandomPolicy(my_grid)
    simulation_object_random = Simulation.Simulation(my_grid, random_policy)
    simulation_object_random.simulate()
    random_policy.print_lookup()
    learned_policy = learnedPolicy.LearnedPolicy(my_grid)
    simulation_object_learned = Simulation.Simulation(my_grid, learned_policy)
    simulation_object_learned.simulate()
    learned_policy.print_lookup()
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Enter question number to run!!!")
    qno = int(input())
    if qno == 1:
        q1()
    elif qno == 2:
        q2()
    elif qno == 3:
        q3()
    elif qno == 4:
        q4()
    elif qno == 5:
        q5()
    else:
        print("Please enter a correct question number to run the program")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
