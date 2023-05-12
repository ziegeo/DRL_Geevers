# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:11:50 2020

@author: LotteH
"""
import torch

ac = torch.load('results/DRL/CBC/Experiment3/model save49999.pt')
ac = ac['structure']
torch.save(ac.state_dict(), 'results/DRL/CBC/Experiment3/model49999.pt')