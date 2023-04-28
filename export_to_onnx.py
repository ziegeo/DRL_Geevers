import pandas as pdmodel
import numpy as np
import torch
from ppo.ppo_network_functions import MLPActorCritic
from inventory_env import InventoryEnv
from torch import nn
import torch.onnx
from cases import General
from ppo.ppo_support_functions import scale_input

#####
experiment_name = 'CBC/PAPERNEWACTIONS/'
case = General()
env = InventoryEnv(case, case.action_low, case.action_high,
                       case.action_min, case.action_max, case.state_low, case.state_high)
####



# ac = torch.load('C:/result_files/CBC/PAPERNEWACTIONS/RN4/pyt_network_save/model save48499 - Copy.pt')
# ac = ac['structure']
# torch.save(ac.state_dict(), 'C:/result_files/CBC/PAPERNEWACTIONS/RN4/pyt_network_save/model save48499 - NEW.pt')


# ac = MLPActorCritic(env.observation_space, env.action_space, env.feasible_actions, (64, 64), nn.Tanh, 0.0,
#                     'uniform')
old = torch.load('C:/result_files/CBC/PAPERNEWACTIONS/RN4/pyt_network_save/model save48499 - Copy.pt')
old = old['structure']
torch.save(old, 'C:/result_files/CBC/PAPERNEWACTIONS/RN4/pyt_network_save/model save48499 - NEW.pt')
model = torch.load('C:/result_files/CBC/PAPERNEWACTIONS/RN4/pyt_network_save/model save48499 - NEW.pt')
print("MODEL")
print(model)
model.eval()

o = env.reset()
o = torch.as_tensor([scale_input(env, o)], dtype = torch.float32)

torch.onnx.export(model,
                  o,
                  'drltest.onnx',
                  training='TrainingMode.EVAL',
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=False)