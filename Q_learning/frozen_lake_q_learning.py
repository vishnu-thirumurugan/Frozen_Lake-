# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 21:22:12 2022

@author: oe21s024
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from q_learning_agent import Agent

if __name__ == '__main__' : 
    env = gym.make('FrozenLake-v1')
    agent = Agent(lr= 0.001, gamma = 0.9, eps_start = 1, eps_end = 0.01, eps_decay = 0.9999995,
                  n_states = 16, n_actions = 4)
    scores = []
    win_percentage = []
    
    n_epsiodes = 500000
    
    for i in range(n_epsiodes):
        done = False
        obs  = env.reset()
        score = 0
        
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info   = env.step(action)
            agent.learn(obs, action, reward, obs_) # updating q values
            score += reward
            obs = obs_
        scores.append(score)
        
        if i % 100 == 0 : # every 100 epsiode, i am updating my everage in that list
            average  = np.mean(scores[-100:])
            win_percentage.append(average)
            
            if i % 1000  == 0 : # to see how epsilon varies for each episode
                print('episode', i, 'win pct %.2f' % average,
                       'epsilon %.2f' % agent.epsilon)
    plt.plot(win_percentage, color = 'orange')
    plt.show()
    
        
            