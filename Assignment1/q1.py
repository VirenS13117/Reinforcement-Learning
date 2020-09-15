from random import randint, random

from main import Action, State, GridBlock


def simulate(my_grid, current_state, action):
    print("simulating")
    decision_prob = random()
    print("decision probability : ", decision_prob)
    if decision_prob < my_grid.prob:
        current_state.move(action.action_list[action.action])
    elif my_grid.prob < decision_prob < my_grid.prob + 0.1:
        current_state.move(action.get_left_perpendicular())
    else:
        current_state.move(action.get_right_perpendicular())

    if my_grid.out_of_bounds(current_state):
        current_state.move(action.get_opposite())
    reward = 0
    if my_grid.is_target(current_state):
        reward = 1
    return current_state, reward


def play_simulation():
    source = (0, 0)
    target = (10, 10)
    my_grid = GridBlock(source, target)
    print("Q1 : Simulate function")
    print("Please enter x coordinate in grid bounds")
    curr_state_x = int(input())  # randint(0, 10)
    while curr_state_x < 0 or curr_state_x > 10:
        print("Please enter x coordinate in grid bounds")
        curr_state_x = int(input())
    print("Please enter y coordinate in grid bounds")
    curr_state_y = int(input())  # randint(0, 10)
    while curr_state_y < 0 or curr_state_y > 10:
        print("Please enter y coordinate in grid bounds")
        curr_state_y = int(input())
    curr_state = State(curr_state_x, curr_state_y)
    i = 0
    while i < 10:
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
