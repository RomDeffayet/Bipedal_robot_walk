# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:04:43 2018

@author: Romain Deffayet
"""
import numpy as np
import gym

ENV_NAME = 'BipedalWalker-v2'


class ARS_agent():
    def __init__(self, env, n_iter = 1000, n_deltas = 16, sigma = 0.03, 
                 n_best = 16, learning_rate = 0.02, max_ep_iter = 2000):
        self.n_iter = n_iter
        self.n_deltas = n_deltas
        self.sigma = sigma
        self.n_best = n_best
        self.learning_rate = learning_rate
        self.max_ep_iter = max_ep_iter
        self.env = gym.make(env)     
        self.normalizer = Normalizer(self.env.observation_space.shape[0])
    
    def train(self):
        theta = np.zeros((self.env.observation_space.shape[0], self.env.action_space.shape[0]))
        for i in range(self.n_iter):
            deltas = self.sample_deltas()
            rewards = self.evaluate(theta, deltas)
            best_rewards = self.keep_best(rewards)
            theta = self.update(theta, best_rewards, rewards, deltas)
        return theta
        
    def sample_deltas(self):
        n, p = (self.env.observation_space.shape[0], self.env.action_space.shape[0])
        deltas = np.zeros((self.n_deltas, n, p))
        for i in range(self.n_deltas):
            deltas[i] = np.random.normal(0, self.sigma, (n,p))
        return deltas
    
    def evaluate(self, theta, deltas = None):
        if deltas is None:
            return self.episode(theta)
        else:
            rewards = np.zeros((self.n_deltas,2))
            for i in range (self.n_deltas):
                r_pos = self.episode(theta + deltas[i])
                r_neg = self.episode(theta - deltas[i])
                rewards[i] = [r_pos,r_neg]
            return rewards
    
    def episode(self, theta, render = False):
        state = self.env.reset()
        total_reward = 0.
        done = False
        count_moves = 0
        while not(done) and count_moves < self.max_ep_iter:
            if (render):
                self.env.render         
            self.normalizer.update_normalizer(state)
            state = self.normalizer.normalize(state)
            action = state.dot(theta)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
        return total_reward
    
    
    def keep_best(self, rewards):
        best_rewards = []
        for [r_pos, r_neg] in rewards:
            best_rewards.append(- max(r_pos, r_neg))        #Minus is to get descneding order
        return np.argsort(best_rewards)[:self.n_best]
        
        
    def update(self, theta, best_rewards, rewards, deltas):
        step = 0.
        for index in best_rewards:
            r_pos, r_neg = rewards[index]
            step += (r_pos - r_neg) * deltas[index]
            
        sigma_rewards = np.std(rewards[:,0] + rewards[:,1])
        theta += self.learning_rate * step / (self.n_best * sigma_rewards)
        
        
        
        
        
class Normalizer():
    
    def __init__(self, nb_inputs):
        self.mean = np.zeros(nb_inputs)
        self.std = np.zeros(nb_inputs)
        self.n = 0
    
    def update_normalizer(self, state):
        last_mean = self.mean.copy()
        last_mean_diff = self.std * self.std * self.n
        
        self.n += 1
        self.mean += (state - last_mean) / self.n
        mean_diff = last_mean_diff + (state - last_mean) * (state - self.mean)
        self.std = np.sqrt(mean_diff / self.n).clip(min = 1e-2)
    
    def normalize(self, state):
        return (state - self.mean) / self.std






def main():
    agent = ARS_agent(ENV_NAME)
    policy = agent.train()
    
    agent.episode(policy, render = True)