# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:04:43 2018

@author: Romain Deffayet
"""
import numpy as np
import gym
from gym import wrappers

ENV_NAME = 'BipedalWalker-v2'
VIDEO_DIR = 'vids'


class ARS_agent():
    def __init__(self, env, n_iter = 1000, n_deltas = 16, sigma = 0.05, 
                 n_best = 16, learning_rate = 0.4, max_ep_iter = 500, monitor_dir = None):
        self.n_iter = n_iter
        self.n_deltas = n_deltas
        self.sigma = sigma
        self.n_best = n_best
        self.learning_rate = learning_rate
        self.max_ep_iter = max_ep_iter
        self.env = gym.make(env)     
        self.normalizer = Normalizer(self.env.observation_space.shape[0])
        if monitor_dir is not None:
            should_record = lambda i: self.record_video
            self.env = wrappers.Monitor(self.env, monitor_dir, video_callable=should_record, force=True)
        self.record_video = False
    
    
    ### The main loop for training. Calls other methods and returns the final policy
    def train(self):
        theta = np.zeros((self.env.observation_space.shape[0], self.env.action_space.shape[0]))
        for i in range(self.n_iter):
            print("Step : ", i, "Reward : ", self.evaluate(theta))
            deltas = self.sample_deltas()
            rewards, rollouts = self.evaluate(theta, deltas)
            sigma_rewards = np.std(rewards[:,0] + rewards[:,1])
            best_rollouts = self.keep_best(rollouts)
            theta = self.update(theta, best_rollouts, sigma_rewards)
        return theta
        
    ### Creating the random samples
    def sample_deltas(self):
        n, p = (self.env.observation_space.shape[0], self.env.action_space.shape[0])
        deltas = np.zeros((self.n_deltas, n, p))
        for i in range(self.n_deltas):
            deltas[i] = np.random.normal(0, self.sigma, (n,p))
        return deltas
    
    ### Returns the reward or the collection of rewards if deltas!=None
    def evaluate(self, theta, deltas = None):
        if deltas is None:
            return self.episode(theta)
        else:
            rewards = np.zeros((self.n_deltas, 2))
            rollouts = []
            for i in range (self.n_deltas):
                r_pos = self.episode(theta + deltas[i])
                r_neg = self.episode(theta - deltas[i])
                rewards[i] = [r_pos, r_neg]
                rollouts.append([r_pos, r_neg, deltas[i]])
            return rewards, rollouts
    
    ### Actually plays an episode and returns the rewards.
    ### Notice that it uses the normalizer class (defined below) to get inputs between -1 and 1
    ### If render==True, it displays the episode on the screen 
    ### If record==True, it records a video and places it in the VIDEO_DIR folder
    def episode(self, theta, render = False, record = False):
        if record:
            self.record_video = True
        state = self.env.reset()
        total_reward = 0.
        done = False
        count_moves = 0
        while not(done) and count_moves < self.max_ep_iter:
            if (render):
                self.env.render()         
            self.normalizer.update_normalizer(state)
            state = self.normalizer.normalize(state)
            action = state.dot(theta)
            state, reward, done, _ = self.env.step(action)
            reward = max(min(reward, 1), -1)
            total_reward += reward
        if (render or record):
            self.env.close()
        self.record_video = False
        return total_reward
    
    ### Only keeps the n_best best improvements (n_best = n_deltas by default)
    def keep_best(self, rollouts):
        best_rollouts = []
        for [r_pos, r_neg, delta] in rollouts:
            best_rollouts.append(- max(r_pos, r_neg))        #Minus is to get descending order
        return [rollouts[k] for k in np.argsort(best_rollouts)[:self.n_best]]
        
    
    ### Does the update step, using the previously defined hyperparameters    
    def update(self, theta, best_rollouts, sigma_rewards):
        step = np.zeros(theta.shape)
        for r_pos, r_neg, delta in best_rollouts:
            step += (r_pos - r_neg) * delta
        theta += self.learning_rate * step / (self.n_best * sigma_rewards)
        return theta
        
        
        
        
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




agent = ARS_agent(ENV_NAME, monitor_dir = VIDEO_DIR)
#policy = agent.train()
#total_reward = agent.episode(policy, render = True)   