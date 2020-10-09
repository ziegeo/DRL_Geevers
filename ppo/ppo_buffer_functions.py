# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:13:38 2020

@author: LotteH

This file contains functions that are relevant for the buffer computations in the PPO algorithm.
"""

import numpy as np
import torch
import math
import random

from ppo.ppo_support_functions import discount_cumsum, statistics_scalar
from torch.utils.data import Dataset

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class Buffer(Dataset):
    '''
    This buffer stores trajectories experienced by the PPO agent who interacts with the environment. 
    Generalized Advantage Estimation (GAE) is used for calculating the advantages of state-action pairs.
    '''

    def __init__(self, obs_dim, act_dim, size, gamma = 0.99, lam = 0.95, cooldown = True):
        '''
        Initializes a buffer to store states, actions, advantages, rewards, returns, values, log probabilities, 
        entropy
        '''
        self.size = size
        self.gamma, self.lam = gamma, lam
        self.cool_down_steps = int(np.rint(math.log(0.5, self.gamma)))
        self.ptr, self.path_start_idx = 0, 0
        self.max_size = self.size + self.cool_down_steps if cooldown else self.size
        self.obs_buf = np.zeros(combined_shape(self.max_size, obs_dim), dtype = np.float32)
        self.act_buf = np.zeros(combined_shape(self.max_size, act_dim), dtype = np.float32)
        self.adv_buf = np.zeros(self.max_size, dtype = np.float32)
        self.rew_buf = np.zeros(self.max_size, dtype = np.float32)
        self.ret_buf = np.zeros(self.max_size, dtype = np.float32)
        self.val_buf = np.zeros(self.max_size, dtype = np.float32)
        self.logp_buf = np.zeros(self.max_size, dtype = np.float32)
        self.entropy_buf = np.zeros(self.max_size, dtype = np.float32)
        # self.entropy_buf = np.zeros(combined_shape(self.max_size, act_dim), dtype = np.float32)
    
    def __len__(self):
        ''' Gets the size of the buffer'''
        return self.size
    
    def __getitem__(self, idx):
        ''' Gets a random sample of the buffer. Returns all different values for the same sample number'''
        sample = {}
        for k, v in self.total_data.items():
            sample[k] = self.total_data[k][idx]
        return sample
    
    def store(self, obs, act, rew, val, logp, entropy):
        '''
        Append one timestep of agent-environment interaction to the buffer.
        '''
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.entropy_buf[self.ptr] = entropy
        self.ptr += 1
        
    def finish_path(self, last_val = 0):
        '''
        Call this at the end of a trajectory. This looks back in the buffer to where the trajectory started, 
        and uses rewards and value estimates from the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as the targets for the value function.
        
        last_val should be V(s_T), the value function estimated for the last state. This allows us to bootstrap
        the reward-to-go calculation to account for timesteps beyond the arbitrary episode horizon.
        '''
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # Implement GAE lambda
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # Compute the reward to go, to be the targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr
    
    def get(self):
        '''
        Call this at the end of an epoch to get all of the data from the buffer, with advantages
        appropriately normalized (mean zero and std 1), and reset some pointers in the buffer
        '''
        assert self.ptr == self.max_size    # buffer has to be full before getting it
        
        # normalize advantage
        adv_mean, adv_std = statistics_scalar(self.adv_buf[:self.size])
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs = torch.as_tensor(self.obs_buf[:self.size], dtype = torch.float32), 
                    act = torch.as_tensor(self.act_buf[:self.size], dtype = torch.float32), 
                    rew = torch.as_tensor(self.rew_buf[:self.size], dtype = torch.float32), 
                    val = torch.as_tensor(self.val_buf[:self.size], dtype = torch.float32), 
                    ret = torch.as_tensor(self.ret_buf[:self.size], dtype = torch.float32), 
                    adv = torch.as_tensor(self.adv_buf[:self.size], dtype = torch.float32), 
                    logp = torch.as_tensor(self.logp_buf[:self.size], dtype = torch.float32), 
                    entropy = torch.as_tensor(self.entropy_buf[:self.size], dtype = torch.float32), 
                    idx = np.arange(self.size))
        self.total_data = data
    
    def get_batches_per_epoch(self, ppo_batch_size):
        '''
        This function returns x training batches per epoch.
        First this was programmed with the DataLoader from pytorch, but this was very time consuming, so 
        shuffling the dictionary ourselves is a better way
        '''
        assert self.size % ppo_batch_size == 0 , 'batch size and buffer size do not match'
        nr_batches = int(self.size / ppo_batch_size)
        indexlist = np.arange(0, self.size)
        random.shuffle(indexlist)
        batches_list = []
        for i in range(nr_batches):
            data_batch = {}
            indexes = indexlist[i * ppo_batch_size : (i + 1) * ppo_batch_size]
            for k, v in self.total_data.items():
                data_batch[k] = self.total_data[k][indexes]
            batches_list.append(data_batch)
        return batches_list
     
    def reset_buffer(self):
        ''' Reset the buffer. '''
        self.ptr, self.path_start_idx = 0, 0
        