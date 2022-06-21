# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 08:15:04 2022

@author: oe21s024
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1')
num_games = 1000
win_pct = []    # every 10 episodes we append the average of scores for 10 episodes
scores = []     # every episode, we have scores

for i in range(num_games):
    done = False # always in the for loop of episodes first set the done flag to false
    obs = env.reset() # you should reset the environment to initial conditions in the second step always
    score = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward
    scores.append(score)
        
    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)
        
plt.plot(win_pct)
plt.show()     
            
        