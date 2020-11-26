from typing import Tuple, List, Set, Dict, Callable
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import trange
import math
import copy
import sys


def e_greedy(Q_value: List, epsilon: float) -> int:
    if np.random.random() <= epsilon:
        action = np.random.choice(np.arange(len(Q_value)))
    else:
        action = np.random.choice(np.where(Q_value == np.max(Q_value))[0])
    return action

class Actions:
    def __init__(self, actions: List[str], moves: defaultdict):
        self.__actions = actions.copy()
        self.__moves = moves.copy()
        self.__action_index, self.__action_value = self.__make_action_index_n_value()
        return

    def __make_action_index_n_value(self) -> Tuple:
        action_index = defaultdict(lambda: -1)
        action_value = defaultdict(lambda: "")
        for k,v in enumerate(self.__actions):
            action_index[v] = k
            action_value[k] = v
        return action_index, action_value

    def actions(self) -> List[str]:
        return self.__actions.copy()

    def n(self) -> int:
        return len(self.__actions)

    def valid(self, action: str) -> bool:
        if action in self.__actions:
            return True
        return False

    def action_index(self, action: str) -> int:
        return self.__action_index[action]

    def action_value(self, index: int) -> str:
        return self.__action_value[index]

    def move(self, action: str) -> Tuple:
        return self.__moves[action]

    def debug(self):
        for a in self.actions():
            i = self.action_index(a)
            v = self.action_value(i)
            m = self.move(a) 
            print("Index: {}, Value: {}, Move: {}.".format(i, v, m))
        return

class Env:
    def __init__(self, 
                 rows: int,  
                 cols: int, 
                 start: Tuple, 
                 goal: Tuple, 
                 actions: Actions,
                 noisy_action: Callable = None, 
                 obstacles: Dict = defaultdict(lambda: set())):
        self.__rows =  rows
        self.__cols = cols
        self.__state = start + tuple()
        self.__path = []
        self.__start = start + tuple()
        self.__goal = goal + tuple()
        self.__obstacles = obstacles.copy()
        self.__actions = copy.deepcopy(actions)
        self.__noisy_action = noisy_action
        return

    def __is_obstacle(self, cell: Tuple, obstacle_type: str) -> bool:
        obstacles = self.__obstacles[obstacle_type]
        if len(obstacles) and cell in obstacles:
            return True
        return False

    def __is_out_of_bounds(self, cell: Tuple) -> bool:
        if cell[0] < 0 or cell[1] < 0 or cell[0] >= self.__rows or cell[1] >= self.__cols:
            return True
        return False

    def __is_goal(self, cell: Tuple) -> bool:
        if cell == self.__goal:
            return True
        return False

    def __is_start(self, cell: Tuple) -> bool:
        if cell == self.__start:
            return True
        return False

    def __take_action(self, state: Tuple, action: str, obstacle_type: str) -> Tuple:
        action =  self.__noisy_action(action) if self.__noisy_action else action
        movement = self.__actions.move(action)
        next_state = (state[0]+movement[0], state[1]+movement[1])
        new_state = state
        if self.is_valid_cell(next_state, obstacle_type):
            new_state = next_state
        self.__path.append(new_state)
        return new_state
    
    def is_valid_cell(self, cell: Tuple, obstacle_type: str) -> bool:
        if self.__is_obstacle(cell, obstacle_type) or self.__is_out_of_bounds(cell):
            return False
        return True 

    def start(self) -> Tuple:
        return self.__start + tuple()

    def goal(self) -> Tuple:
        return self.__goal + tuple()

    def obstacles(self, obstacle_type: str = "default") -> List:
        obs = self.__obstacles[obstacle_type]
        return obs.copy()
    
    def reset(self, obstacle_type: str = "default", is_random: bool = False, is_non_terminal: bool = True) -> Tuple:
        self.__path.clear()
        self.__state = self.__start + tuple()
        if is_random:
            while True:
                r = np.random.choice(np.arange(self.__rows))
                c = np.random.choice(np.arange(self.__cols))
                if self.is_valid_cell((r,c), obstacle_type):
                    if is_non_terminal:
                        if not self.__is_goal((r,c)):
                            self.__state = (r,c)
                            break
                    else:
                        self.__state = (r,c)
                        break

        return self.__state

    def actions(self) -> Actions:
        return self.__actions

    def reward(self, cell: Tuple) -> int:
        if self.__is_goal(cell):
            return 1
        return 0

    def rows(self) -> int:
        return self.__rows
    
    def cols(self) -> int:
        return self.__cols

    def step(self, action: str, obstacle_type: str = "default", is_simulate: bool = False) -> Tuple:
        next_state = self.__state + tuple()
        done = False
        if self.__is_goal(self.__state):
            done = True
        else:
            next_state = self.__take_action(self.__state, action, obstacle_type)
            if self.__is_goal(next_state):
                done = True
            if not is_simulate:
                self.__state = next_state + tuple()

        return (next_state, self.reward(self.__state), done, {})

    def display(self, obstacle_type: str = "default") -> np.ndarray:
        m = []
        for i in range(self.__rows-1, 0-1, -1):
            n = []
            for j in range(self.__cols):
                if self.__is_goal((i,j)):
                    n.append("G")
                elif self.__is_start((i,j)):
                    n.append("S")
                elif (i,j) in self.__path:
                    n.append("*")
                elif not self.is_valid_cell((i,j), obstacle_type):
                    n.append("#") 
                else:
                    n.append(".")
            m.append(n)
        m = np.array(m)
        return m

"""**Make environment**"""

def make_env(goal: Tuple = (10,10)) -> Env:
    # Define environment
    start = (0,0)
    end = (10,10)
    obstacles = set([(0,5), (2,5), (3,5), (4,5), (5,5), (6,5), (7,5), (9,5), (10,5),
                (5,0), (5,2), (5,3), (5,4),
                (4,6), (4,7), (4,9), (4,10)])

    if goal == start:
        print("The goal " + str(goal) + " is the start position itself hence agent have no work to do ")
    
    if goal in obstacles:
        print("The goal " + str(goal) + " is one of the obstacle and hence is an unreachable goal ")
    
    if not (start[0]  <= goal[0] <= end[0]) or not (start[1] <= goal[1] <= end[1]):
        print("The goal " + str(goal) + " is out of bound and hence uncreachable goal ")

    action_list = ["right", "up", "left", "down"]
    action_moves = defaultdict(lambda: (0,0), {"right": (0,1), "up": (1,0), "left": (0,-1), "down": (-1,0)})
    actions = Actions(actions=action_list, moves=action_moves)
    
    def noisy_action(action: str = "", threshold: float = 0.8) -> str:
        
        if action == "left" or action == "right":
            if np.random.random() <= threshold:
                return action
            else:
                return np.random.choice(["up", "down"])
        
        if action == "up" or action == "down":
            if np.random.random() <= threshold:
                return action
            else:
               return np.random.choice(["left", "right"])
        
        return action


    obs = defaultdict(lambda: set())
    obs["default"] = obstacles

    env = Env(rows=11, cols=11,
              start=start, goal=goal, 
              actions=actions, noisy_action=noisy_action, 
              obstacles=obs)

    return env

env_default = make_env()



def plot_performance(steps: np.ndarray, labels: List[str], clrs: List[str], title: str):
    fig, s_plt = plt.subplots(1, figsize=(10, 6), facecolor="white")
    ydata = [y for y in range(steps.shape[2])]
    for i in range(steps.shape[0]):
        xdata = np.average(steps[i], axis=0)
        s_plt.plot(xdata, ydata, color=clrs[i], label=labels[i])
        xstderr = np.std(steps[i], axis=0)
        xstderr *= 1 / math.sqrt(steps.shape[1])
        xstderr *= 1.96
        s_plt.fill_betweenx(ydata, np.subtract(xdata, xstderr), np.add(xdata, xstderr),  alpha=0.2, color=clrs[i])
    plt.legend(loc="upper left")
    plt.xlabel("Time Steps")
    plt.ylabel("Episodes")
    plt.title(title + ":" + " Episodes = " + str(steps.shape[2]) + "," + " Trials = " + str(steps.shape[1]))
    plt.show()
    return


def groups(dim: int, step: int) -> dict:
    G = dict()
    count = 0
    for i in range(0, dim, step):
        for j in range(0, dim, step):
            for r in range(i,i+step):
                for c in range(j,j+step): 
                    G[(r,c)] = count
            count += 1

    G["nums"] = int(math.ceil(dim/step))**2
    print("Groups: {}".format(str(G["nums"])))
    
    m = []
    for r in range(dim):
        n = []
        for c in range(dim):
            n.append(G[(r,c)])
        m.append(n)
    m = np.array(m)
    return G

"""**SARSA SEMI GRADIENT**"""

def sarsa_semi_gradient(env: Env, 
          pi: Callable, 
          trials: int, 
          episodes: int, 
          alpha: float, 
          gamma: float, 
          epsilon: float,
          Groups: Dict) -> np.ndarray:

    n = env.actions().n()
    num_features = n * Groups["nums"]

    grads = np.zeros(num_features)

    def grad(state: Tuple, offset: int) -> np.ndarray:
        f = grads.copy()
        f[Groups[state] * n + offset] = 1
        return f

    def q_value(W: np.ndarray, state: Tuple) -> np.ndarray:
        offset = Groups[state] * n
        q = W[offset: (offset + n)]  
        return q

    trials_steps = []
    for t in trange(trials, desc="Trials"):

        steps = []
        cum_steps = 0
        W = np.zeros(num_features)
        
        for e in trange(episodes, desc="Episodes", leave=False):
            state = env.reset()
            action = pi(q_value(W, state), epsilon)
            done = False
            count = 0
            while not done:
                state_next, reward, done, _ = env.step(env.actions().action_value(action))
                action_next = pi(q_value(W, state_next), epsilon)
                state_offset = Groups[state] * n + action
                state_next_offset = Groups[state_next] * n + action_next
                if done:
                    err = (reward - W[state_offset])
                else:
                    err = (reward + gamma * W[state_next_offset] - W[state_offset])

                gradient = grad(state, action)

                dW = alpha * err * gradient
                W = np.add(W, dW)
                state = state_next
                action = action_next
                count = count + 1
            
            cum_steps += count
            steps.append(cum_steps)
            
        trials_steps.append(steps)

    return np.array(trials_steps)


def sarsa_semi_gradient_ext(env: Env, 
          pi: Callable, 
          trials: int, 
          episodes: int, 
          alpha: float, 
          gamma: float, 
          epsilon: float) -> np.ndarray:

    n = env.actions().n() + 3
    num_features = n * env.rows() * env.cols()

    grads = np.array([1 if ((i+1)%n == 0) else 0 for i in range(num_features)])

    def grad(state: Tuple, action: int) -> np.ndarray:
        f = grads.copy()
        offset = (state[0] * env.cols() + state[1]) * n
        f[offset + action] = 1
        f[offset + 4] = state[0]
        f[offset + 5] = state[1]
        return f

    def q_value(W: np.ndarray, state: Tuple) -> np.ndarray:
        offset = (state[0] * env.cols() + state[1]) * n 
        q = W[offset: (offset + 4)]  
        return q

    trials_steps = []
    for t in trange(trials, desc="Trials"):

        steps = []
        cum_steps = 0
        W = np.zeros(num_features)
        
        for e in trange(episodes, desc="Episodes", leave=False):
            state = env.reset()
            action = pi(q_value(W, state), epsilon)
            done = False
            count = 0
            while not done:
                state_next, reward, done, _ = env.step(env.actions().action_value(action))
                action_next = pi(q_value(W, state_next), epsilon)
                state_offset = ((state[0] * env.cols() + state[1]) * n)  + action
                state_next_offset = ((state_next[0] * env.cols() + state_next[1]) * n) + action_next
                if done:
                    err = (reward - W[state_offset])
                else:
                    err = (reward + gamma * W[state_next_offset] - W[state_offset])
                gradient = grad(state, action)
                dW = alpha * err * gradient
                W = np.add(W, dW)
                state = state_next
                action = action_next
                count = count + 1
            cum_steps += count
            steps.append(cum_steps)
        trials_steps.append(steps)
    return np.array(trials_steps)


def sarsa_semi_gradient_ext_10(env: Env, 
          pi: Callable, 
          trials: int, 
          episodes: int, 
          alpha: float, 
          gamma: float, 
          epsilon: float) -> np.ndarray:

    n = env.actions().n() + 3 + 3
    num_features = n * env.rows() * env.cols()

    grads = np.array([1 if ((i+1) % n == 0) else 0 for i in range(num_features)])

    def quadrant(state: Tuple) -> Tuple:
        if 0 <= state[0] < 5:
            if 5 <= state[1]:
                return (0,1)
            else:
                return (0,0)
        else:
            if 5 <= state[1]:
                return (1,0)
            else:
                return (1,1)
        return (0,0)

    def grad(state: Tuple, action: int) -> np.ndarray:
        f = grads.copy()
        offset = (state[0] * env.cols() + state[1]) * n
        qd = quadrant(state)
        f[offset + action] = 1
        f[offset + 4] = state[0]
        f[offset + 5] = state[1]
        f[offset + 6] = qd[0]
        f[offset + 7] = qd[1]
        f[offset + 8] = 1 if state != (10,10) else 0
        return f

    def q_value(W: np.ndarray, state: Tuple) -> np.ndarray:
        offset = (state[0] * env.cols() + state[1]) * n 
        q = W[offset: (offset + 4)]  
        return q

    trials_steps = []
    for t in trange(trials, desc="Trials"):

        steps = []
        cum_steps = 0
        W = np.zeros(num_features)
        
        for e in trange(episodes, desc="Episodes", leave=False):
            state = env.reset()
            action = pi(q_value(W, state), epsilon)
            done = False
            count = 0
            while not done:
                state_next, reward, done, _ = env.step(env.actions().action_value(action))
                action_next = pi(q_value(W, state_next), epsilon)
                state_offset = ((state[0] * env.cols() + state[1]) * n)  + action
                state_next_offset = ((state_next[0] * env.cols() + state_next[1]) * n) + action_next
                if done:
                    err = (reward - W[state_offset])
                else:
                    err = (reward + gamma * W[state_next_offset] - W[state_offset])

                gradient = grad(state, action)
                dW = alpha * err * gradient
                W = np.add(W, dW)
                state = state_next
                action = action_next
                count = count + 1
            
            cum_steps += count
            steps.append(cum_steps)
            
        trials_steps.append(steps)

    return np.array(trials_steps)



def Q3b(ts_ssg_1):
    steps = []
    steps.append(ts_ssg_1)
    steps = np.array(steps)
    plot_performance(steps, labels=["Semi Gradient Sarsa(1x1)"], clrs=["red"], title="Semi Gradient One Step Sarsa")
    return

def Q3c(ts_ssg_1, ts_ssg_2, ts_ssg_3, ts_ssg_4):
    steps = []
    steps.append(ts_ssg_1)
    steps.append(ts_ssg_2)
    steps.append(ts_ssg_3)
    steps = np.array(steps)
    plot_performance(steps, labels=["Sarsa state aggregation (1x1)", "Sarsa state aggregation (2x2)", "Sarsa state aggregation (3x3)"],
                     clrs=["red", "blue", "black"],
                     title="Semi Gradient One Step Sarsa with state aggregations with 1,2,3 groups")

    steps = []
    steps.append(ts_ssg_1)
    steps.append(ts_ssg_4)
    steps = np.array(steps)
    plot_performance(steps, labels=["Sarsa Semi Gradient (1x1)", "Sarsa Semi Gradient (4x4)"], clrs=["red", "blue"], title="Semi Gradient One Step Sarsa with state aggregations 1 and 4 groups)")
    return

def Q3d(ts_ssg_1, ts_ssg_e_1):
    steps = []
    steps.append(ts_ssg_1)
    steps.append(ts_ssg_e_1)
    steps = np.array(steps)
    plot_performance(steps,
                     labels=["Saras Semi Gradient", "Saras Semi Gradient Exended Features"],
                     clrs=["darkblue", "crimson"],
                     title="Sarsa Semi Gradient")
    return

def Q3e(ts_ssg_1, ts_ssg_e_1, ts_ssg_e_1_10):
    steps = []
    steps.append(ts_ssg_1)
    steps.append(ts_ssg_e_1)
    steps.append(ts_ssg_e_1_10)
    steps = np.array(steps)
    plot_performance(steps, labels=["Saras Semi Gradient", "Saras Semi Gradient Exended Features", "Saras Semi Gradient Exended Features 10"],
                     clrs=["darkblue", "crimson", "forestgreen"],
                     title="Sarsa Semi Gradient")
    return

if __name__=="__main__":
    ts_ssg_1 = sarsa_semi_gradient(env=env_default, pi=e_greedy, trials=10, episodes=100, alpha=0.1, gamma=1, epsilon=0.1, Groups=groups(11, 1))
    ts_ssg_2 = sarsa_semi_gradient(env=env_default, pi=e_greedy, trials=10, episodes=100, alpha=0.1, gamma=1, epsilon=0.1, Groups=groups(11, 2))
    ts_ssg_3 = sarsa_semi_gradient(env=env_default, pi=e_greedy, trials=10, episodes=100, alpha=0.1, gamma=1, epsilon=0.1, Groups=groups(11, 3))
    ts_ssg_4 = sarsa_semi_gradient(env=env_default, pi=e_greedy, trials=10, episodes=100, alpha=0.1, gamma=1, epsilon=0.1, Groups=groups(11, 4))
    ts_ssg_e_1_10 = sarsa_semi_gradient_ext_10(env=env_default, pi=e_greedy, trials=10, episodes=100, alpha=0.1, gamma=1, epsilon=0.1)
    ts_ssg_e_1 = sarsa_semi_gradient_ext(env=env_default, pi=e_greedy, trials=10, episodes=100, alpha=0.1, gamma=1, epsilon=0.1)
    # Q3b(ts_ssg_1)
    Q3c(ts_ssg_1, ts_ssg_2, ts_ssg_3, ts_ssg_4)
    Q3d(ts_ssg_1, ts_ssg_e_1)
    Q3e(ts_ssg_1, ts_ssg_e_1, ts_ssg_e_1_10)
