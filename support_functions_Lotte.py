# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:19:47 2020

@author: KevinG
"""

import numpy as np
import random
import torch
import torch.nn as nn
import scipy.signal


def statistics_scalar(x, with_min_max=False):
    '''
    Get mean and max of an array
    '''
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = sum(x), len(x)
    mean = global_sum / global_n

    global_sum_sq = (np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)

    if with_min_max:
        global_min = np.min(x)
        global_max = np.max(x)
        return mean, std, global_min, global_max

    return mean, std


def set_seeds(env, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.action_space.seed(seed)


def mlp(sizes, activation, output_activation=nn.Identity):
    '''
    This function creates a neural network with (sizes) layers (1 input, x hidden layers, 1 output layer)
    - sizes:                nr of total layers
    - activation:           activation function of hidden layers
    - output_activation:    activation function of output layer. Default is no activation.
    '''
    layers = []
    for j in range(len(sizes)-1):
        if j < len(sizes) - 2:
            # for all the hidden layers
            act = activation()
        elif output_activation == nn.Softmax:
            # for the output layer of the actor network
            act = output_activation(0)
        else:
            # for the output layer of the critic network
            act = output_activation()
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]
    return nn.Sequential(*layers)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def standardize_input(env, state):
    '''
    Standardize the input data, and use a maximum of 5 on both ends to ensure
    that training isn't going terrible.
    '''
    assert [(env.benchmark_std_states[i] == 0) for i in range(env.products)], \
                'Benchmark policy of environment not computed correctly'
    standardized_input = (state - env.benchmark_mean_states) / (env.benchmark_std_states)
    return np.maximum(-5.0, np.minimum(5.0, standardized_input))


def scale_input(env, state):
    """
    Scale the input data, such that it is between -1 and 1
    """
    scaled_input = state / env.observation_space.high
    return scaled_input
