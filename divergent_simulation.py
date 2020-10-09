"""@author: KevinG."""
import numpy as np
from inventory_env import InventoryEnv
from cases import Divergent
import random

experiment_name                 = 'Divergent'

case                            = Divergent()
case.order_policy               = 'BaseStock'                           # Predetermined order policy, can be either 'X','X+Y' or 'BaseStock'
action_low                      = np.array([-1,-1,-1,-1])
action_high                     = np.array([1,1,1,1])
action_min                      = np.array([0,0,0,0])
action_max                      = np.array([150,150,150,150])
state_low                       = np.zeros(21)
state_high                      = np.array([8000, 6000,             # Total inventory and total backorders
                                            2000,2000,2000,2000,    # Inventory per stockpoint
                                            2000,2000,2000,         # Backorders per stockpoint
                                            250,250,250,250,        # Demand of previous period
                                            250,250,250,250,        # In transit
                                            250,250,250,250])       # transport
horizon                         = 2
simulation_horizon              = 75


simulation_length           = 75                            	# length of simulation to compute benchmark and as stopping criterion
warmup_period               = 25  

totaltotalreward = 0
for k in range(20):
    # print("Replication " + str(k))
    # Initialize environment
    env = InventoryEnv(case, horizon, action_low, action_high,
                       action_min, action_max, state_low, state_high, 'Simulation',
                       "Continuous", fix=True)
    run_name = "RN{}".format(k)
    random.seed(k)
    np.random.seed(k)

    totalreward, totalholdingcosts, totalbackordercosts = 0, 0, 0
    _ = env.reset()
    for t in range(simulation_length):
        _, reward, _, info = env.simulate([116, 23, 23, 23],False)
        if t >= warmup_period:
            totalholdingcosts += info['holding_costs']
            totalbackordercosts += info['backorder_costs']
            totalreward += reward
    totaltotalreward += totalreward
    print("Total reward: {}, Total holding costs: {}, Total backordercosts {}".format(totalreward,totalholdingcosts, totalbackordercosts))
print(totaltotalreward/20)