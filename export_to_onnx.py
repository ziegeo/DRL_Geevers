import pandas as pdmodel
import numpy as np
import torch
from torch import nn
import torch.onnx

# model = torch.load('D:/KevinG/OneDrive/OneDrive - Ortec B.V/DRLinIM/Reinforcement-Learning-in-Inventory-Management/results/DRL/CBC/Experiment5/model save29999 (1).pt')
model = torch.load('D:/KevinG/OneDrive/OneDrive - Ortec B.V/DRLinIM/Reinforcement-Learning-in-Inventory-Management/results/DRL/BeerGame/TRUE/model save24999 (0).pt')
print(model)
model.eval()
low_actions = np.zeros([50])

torch.onnx.export(model,
                  low_actions,
                  'drltest.onnx',
                  training='TrainingMode.PRESERVE',
                  export_params=True,
                  opset_version=10)