from random import randint, random

from Assignment1.environment import Action, State, Environment


def simulate(my_grid, current_state, action):
    print("simulating")
    decision_prob = random()
    print("decision probability : ", decision_prob)
    if decision_prob < my_grid.prob:
        current_state.move(action.action_list[action.action])
    elif my_grid.prob < decision_prob < my_grid.prob + (1 - my_grid.prob) / 2:
        current_state.move(action.get_left_perpendicular())
    else:
        current_state.move(action.get_right_perpendicular())

    if my_grid.out_of_bounds(current_state):
        print("out of bound")
        current_state.move(action.get_opposite())
    reward = 0
    if my_grid.is_target(current_state):
        reward = 1
    return current_state, reward


def play_simulation():
    source = (0, 0)
    target = (10, 10)
    my_grid = Environment(source, target)
    print("Q1 : Simulate function")
    curr_state = State(source[0], source[1])
    i = 0
    while i < 100:
        print("current state : ", curr_state)
        print("Choose one of the following actions : ")
        actions_available = ['up', 'left', 'down', 'right']
        print(actions_available)
        action = input()
        while action not in actions_available:
            print("invalid action : ")
            action = input()
        action_object = Action(action)
        [next_state, reward] = simulate(my_grid, curr_state, action_object)
        print("next state : ", next_state.x, " ", next_state.y)
        print("reward obtained : ", reward)
        print("ending simulation")
        curr_state = next_state
        i += 1
