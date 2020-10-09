# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:11:50 2020

@author: LotteH
"""
import torch

ac = torch.load('results/DRL/Divergent/model save1999.pt')
ac = ac['structure']
torch.save(ac.state_dict(), 'results/DRL/Divergent/newresult1999.pt')

# ac2 = torch.load('results/Divergent/model save0.pt')
# ac2 = ac2['structure']
# torch.save(ac2.state_dict(), 'results/Divergent/newresult0.pt')