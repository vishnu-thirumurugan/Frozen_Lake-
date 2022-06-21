# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:21:50 2022

@author: oe21s024
"""

import numpy as np

# defing agent
class Agent():
    
    def __init__(self, lr, gamma, n_actions, n_states, eps_start, eps_end, eps_decay):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.Q = {}
        self.init_Q()
        
    def init_Q(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q[(state, action)] = 0
        
    def choose_action (self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice([i for i in range(self.n_actions)])
        else:
            actions = np.array([self.Q[(state, a)] for a in range(self.n_actions)])
            action  = np.argmax(actions)
        return action

    def epsilon_decrement (self):
        if self.epsilon > self.eps_end:
            self.epsilon = self.epsilon * self.eps_decay
        else:
            self.epsilon = self.eps_end
    
    # the following function is where the q values get updated
    def learn (self,state, action, reward, state_):
        actions = np.array([self.Q[(state_,a)] for a in range(self.n_actions)])
        a_max  = np.argmax(actions)
        
        self.Q[(state, action)] += self.lr*(reward + self.gamma*self.Q[(state_, a_max)] - self.Q[(state, action)])
        
        #after updating q value, update ur epsilon value here itself
        #it makes your code more object oriented
        self.epsilon_decrement()
        
        
        
        
