"""@author: KevinG."""
import numpy as np
from inventory_env_old import InventoryEnv
from cases import Divergent
import random

case               = Divergent()
case.order_policy  = 'BaseStock'                           # Predetermined order policy, can be either 'X','X+Y' or 'BaseStock' 
action             = [[124, 30, 30, 30]]
replications       = 1000

for i in range(len(action)):
    totaltotalreward = 0
    rewardlist = []
    mintotalreward = -1000000
    for k in range(replications):
        env = InventoryEnv(case, case.action_low, case.action_high, case.action_min,
                        case.action_max, case.state_low, case.state_high, 'Simulation')
        run_name = "RN{}".format(k)
        random.seed(k)
        np.random.seed(k)
        totalreward, totalholdingcosts, totalbackordercosts = 0, 0, 0
        _ = env.reset()
        for t in range(case.horizon):
            _, reward, _, info = env.simulate(action[i],False)
            if t >= case.warmup:
                totalholdingcosts += info['holding_costs']
                totalbackordercosts += info['backorder_costs']
                totalreward += reward
        if totalreward > mintotalreward:
            mintotalreward = totalreward
        totaltotalreward += totalreward
        rewardlist.append(totalreward)
    print("Average total reward: {}, Lowest total reward:{}, Action: {}, Complete rewardlist: {}".format(totaltotalreward/replications, mintotalreward, action[i], rewardlist))