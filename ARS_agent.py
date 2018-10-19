# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:04:43 2018

@author: Romain Deffayet
"""

class ARS_agent():
    def __init__(self, n_iter, n_deltas, sigma, n_best, learning_rate, env, max_ep_iter):
        self.n_iter = n_iter
        self.n_deltas = n_deltas
        self.sigma = sigma
        self.n_best = n_best
        self.learning_rate = learning_rate
        self.env = gym.make(env)
        self.max_ep_iter = max_ep_iter
    
    def train(self):
        theta = np.zeros(np.size())
        for i in range(self.n_iter):
            deltas = self.sample_deltas()
            rewards = self.evaluate(theta, deltas)
            best_rewards = self.keep_best(rewards)
            theta = improve(theta, best_rewards, self.learning_rate)
        return theta
        
    def sample_deltas(self):
        n, p = np.size()
        deltas = np.zeros((self.n_deltas, n, p))
        for delta in range(self.n_deltas):
            deltas[delta] = np.random.norm((n,p), 0, self.sigma)
        return deltas
    
    def evaluate(self, theta, deltas = None):
        if (deltas == None):
            return self.episode(theta)
        else:
            rewards = []
            for i in range (np.size(deltas)[1]):
                r_pos = self.episode(theta + deltas[i])
                r_neg = self.episode(theta - deltas[i])
                rewards.append([r_pos,r_neg])
            return rewards
    
    def episode(self, theta):
        state = self.env.reset()
        total_reward = 0.
        done = False
        count_moves = 0
        while not(done) and count_moves < self.max_ep_iter:
            action = self.theta.dot(state)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
        return total_reward
        
    
    def keep_best(self, rewards):
        
        
        
    def improve(self, ...):
        