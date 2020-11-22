import numpy as np
from collections import defaultdict
from Ex6.src.Environment import Grid
import random

class Stochastic_Model:
    def __init__(self, env):
        self.transitions = {}
        self.env = env

    def add(self, s, a, s_prime):
        self.transitions[(s,a)] = s_prime

    def sample(self):
        l = [i for i in self.transitions]
        (s,a) = random.choice(l)
        return s, a

    def step(self, s, a):
        if (s,a) not in self.transitions:
            s,r,done,_ = self.env.make_transition(s,a)
            return s,r,0
        s_prime = self.transitions[(s, a)]
        possible_states = [(i[0],i[1]) for i in s_prime]
        probs = [i[2] for i in s_prime]
        index = np.random.choice(len(possible_states), p=probs)
        s_new = possible_states[index]
        return s_new[0], s_new[1]
