# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 09:09:02 2018

@author: asus
"""

import gym
env = gym.make('BipedalWalker-v2')
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action