# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:11:55 2020

@author: KevinG
"""
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions.categorical import Categorical
from support_functions_Lotte import mlp


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        '''
        Produce action distributions for given observations, and optionally
        compute the log likelihood of given actions under those distributions.
        '''
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPDiscreteActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        '''
        Initialize the actor with discrete actions.
        Output activation function = Softmax.
        '''
        super().__init__()
        self.pi_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim],
                          activation)

    def _distribution(self, obs):
        logits = self.pi_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        '''
        Initialize the critic.
        No output activation function.
        '''
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # making sure that v has the right shape
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(64, 64),
                 activation=nn.Tanh):
        super().__init__()
        obs_dim = observation_space.shape[0]

        # Initially just focus on discrete action space
        self.pi = MLPDiscreteActor(obs_dim, action_space.n, hidden_sizes,
                                   activation)
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

        # Initialize weights
        for i in [0, 2, 4]:
            init.constant_(self.v.v_net[i].bias, 0.0)
            init.constant_(self.pi.pi_net[i].bias, 0.0)
            if activation == nn.Tanh:
                init.xavier_normal_(self.v.v_net[i].weight, gain=5/3)
                init.xavier_normal_(self.pi.pi_net[i].weight, gain=5/3)
            else:
                init.xavier_normal_(self.v.v_net[i].weight, gain=math.sqrt(2))
                init.xavier_normal_(self.pi.pi_net[i].weight,
                                    gain=math.sqrt(2))

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            entropy = pi.entropy()
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy(), entropy.numpy()

    def get_action_probs(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
        return pi.probs.numpy()

    def act(self, obs):
        return self.step(obs)[0]
