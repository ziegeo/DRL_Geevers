# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:06:25 2020

@author: KevinG
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:13:38 2020

@author: LotteH
"""

import numpy as np
import torch
import math

from support_functions_Lotte import discount_cumsum, statistics_scalar
from torch.utils.data import Dataset

class Buffer(Dataset):
    '''
    This buffer stores trajectories experienced by the PPO agent who interacts with the environment. 
    Generalized Advantage Estimation (GAE) is used for calculating the advantages of state-action pairs.
    '''
    
    def __init__(self, obs_dim, act_dim, size, gamma = 0.99, lam = 1, cooldown = True):
        self.size = size
        self.gamma, self.lam = gamma, lam
        self.cool_down_steps = int(np.rint(math.log(0.5, self.gamma)))
        self.ptr, self.path_start_idx = 0, 0
        self.max_size = self.size + self.cool_down_steps if cooldown else self.size
        self.obs_buf = np.zeros((self.max_size, obs_dim), dtype = np.float32)
        self.act_buf = np.zeros(self.max_size, dtype = np.float32)
        self.adv_buf = np.zeros(self.max_size, dtype = np.float32)
        self.rew_buf = np.zeros(self.max_size, dtype = np.float32)
        self.ret_buf = np.zeros(self.max_size, dtype = np.float32)
        self.val_buf = np.zeros(self.max_size, dtype = np.float32)
        self.logp_buf = np.zeros(self.max_size, dtype = np.float32)
        self.entropy_buf = np.zeros(self.max_size, dtype = np.float32)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        sample = {}
        for k, v in self.total_data.items():
            sample[k] = torch.as_tensor(self.total_data[k][idx], dtype = torch.float32)
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
        data = dict(obs = self.obs_buf[:self.size], act = self.act_buf[:self.size], 
                    rew = self.rew_buf[:self.size], val = self.val_buf[:self.size], 
                    ret = self.ret_buf[:self.size], adv = self.adv_buf[:self.size], 
                    logp = self.logp_buf[:self.size], entropy = self.entropy_buf[:self.size], 
                    idx = np.arange(self.size))
        self.total_data = data
    
    def reset_buffer(self):
        self.ptr, self.path_start_idx = 0, 0