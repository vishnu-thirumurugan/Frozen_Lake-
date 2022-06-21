# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 08:58:15 2022

@author: oe21s024
"""
# required dependencies
import gym
import numpy as np
import matplotlib.pyplot as plt

# making environment 
env = gym.make('FrozenLake-v1')
 
 #SFFF ---> 0 1 2 3
 #FHFH ---> 4 5 6 7
 #FFFH ---> 8 9 10 11
 #HFFG ---> 12 13 14 15
 
 # initializing parameters
num_games = 1000
win_pct = []
scores  = []
 
 # define policy  ---> deterministic
policy = {0 :1, 1:2, 2:1, 3:0, 4:1, 6:1, 8:2, 9:1, 10:1, 13:2, 14:2 }
 
 # training
for i in range (num_games):
     done = False
     obs  = env.reset()
     score = 0
     
     while not done:
         action = policy[obs]
         obs, reward, done, info = env.step(action)
         score += reward
     scores.append(score)
    
     if i % 10 == 0 :
         average = np.mean(scores[-10:])
         win_pct.append(average)
plt.plot(win_pct)
plt.show()
         