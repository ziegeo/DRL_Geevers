# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:11:50 2020

@author: LotteH
"""
import torch

for i in range(1, 11):
    ac = torch.load('results/DRL/CBC/Experiment5/model save29999 ({}).pt'.format(i))
    ac = ac['structure']
    torch.save(ac.state_dict(), 'results/DRL/CBC/Experiment5/newresult29999RN{}.pt'.format(i-1))