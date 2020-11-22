import numpy as np
from collections import defaultdict
from Ex6.src.Environment import Grid
import random

class Model:
    def __init__(self, env, n_states, n_actions):
        self.transitions = {}
        self.rewards = {}
        self.time = {}
        self.env = env

    def add(self, s, a, s_prime, r, time = 0):
        self.transitions[(s,a)] = s_prime
        self.rewards[(s,a)] = r
        self.time[(s,a)] = time

    def sample(self):
        (s,a) = random.choice(list(self.transitions))
        return s, a

    def step(self, s, a):
        # print(self.transitions)
        if (s,a) not in self.transitions:
            s,r,done,_ = self.env.make_transition(s,a)
            return s,r,0
        s_prime = self.transitions[(s,a)]
        r = self.rewards[(s,a)]
        time_step = self.time[(s,a)]
        return s_prime, r, time_step
