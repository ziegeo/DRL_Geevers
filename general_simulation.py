"""@author: KevinG."""
import numpy as np
import random
from inventory_env import InventoryEnv
from cases import General


case               = General()
case.order_policy  = 'BaseStock'                           # Predetermined order policy, can be either 'X','X+Y' or 'BaseStock'
action             = [[82, 100, 64, 83, 35, 35, 35, 35, 35]]
replications       = 250

for i in range(len(action)):
    totaltotalreward = 0
    totalfulfilled = np.zeros([case.no_nodes, case.no_nodes], dtype=int)
    totaldemand = np.zeros([case.no_nodes, case.no_nodes], dtype=int)
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
        totaltotalreward += totalreward
        totalfulfilled += env.TotalFulfilled
        totaldemand += env.TotalDemand
    fillrate = totalfulfilled/totaldemand
    print(fillrate)
    print("Average total reward: {}, Action: {}".format(totaltotalreward/replications, action[i]))