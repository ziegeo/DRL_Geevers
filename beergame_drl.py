"""@author: KevinG."""
import numpy as np
from inventory_env import InventoryEnv
# import spinup
# from spinup import ppo_pytorch as ppo
from AC_Lotte import MLPActorCritic
# import torch
import torch.nn as nn
from ppo_torch_training_dataset_Lotte import ppo_learning
from support_functions_Lotte import set_seeds


def encode_state(state):
    """Encode the state, so we can find it in the q_table."""
    encoded_state = (state[0] - 1) * 729
    encoded_state += (state[1] - 1) * 81
    encoded_state += (state[2] - 1) * 9
    encoded_state += (state[3] - 1)
    return int(encoded_state)

def encode_action(action):
    """Encode the action, so we can find it in the q_table."""
    encoded_action = action[0] * 64
    encoded_action += action[1] * 16
    encoded_action += action[2] * 4
    encoded_action += action[3]
    return int(encoded_action)

class BeerGame:
    """Based on the beer game by Chaharsooghi (2008)."""

    def __init__(self, timesteps):
        # Supply chain variables
        # Number of nodes per echelon, including suppliers and customers
        # The first element is the number of suppliers
        # The last element is the number of customers
        stockpoints_echelon = [1, 1, 1, 1, 1, 1]
        # Number of suppliers
        no_suppliers = stockpoints_echelon[0]
        # Number of customers
        no_customers = stockpoints_echelon[-1]
        # Number of stockpoints
        no_stockpoints = sum(stockpoints_echelon) - \
            no_suppliers - no_customers

        # Connections between every stockpoint
        connections = np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0]
            ])

        # Unsatisfied demand
        # This can be either 'backorders' or 'lost_sales'
        unsatisfied_demand = 'backorders'

        # Goal of the method
        # This can be either 'target_service_level' or 'minimize_costs'
        goal = 'minimize_costs'
        # Target service level, required if goal is 'target_service_level'
        tsl = 0.95
        # Costs, required if goal is 'minimize_costs'
        holding_costs = [0, 1, 1, 1, 1, 0]
        bo_costs = [2, 2, 2, 2, 2, 2]

        # order_policy = 'X+Y'
        demand_dist = 'normal'
        demand_lb = 0
        demand_ub = 15
        leadtime_dist = 'normal'
        leadtime_lb = 0
        leadtime_ub = 4

        # State-Action variables
        possible_states = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        possible_actions = [0, 1, 2, 3]

        no_states = len(possible_states) ** no_stockpoints
        no_actions = len(possible_actions) ** no_stockpoints
        # Initialize environment
        self.env = InventoryEnv(stockpoints_echelon=stockpoints_echelon,
                                no_suppliers=no_suppliers,
                                no_customers=no_customers,
                                no_stockpoints=no_stockpoints,
                                connections=connections,
                                unsatisfied_demand=unsatisfied_demand,
                                goal=goal,
                                tsl=tsl,
                                holding_costs=holding_costs,
                                bo_costs=bo_costs,
                                initial_inventory=12,
                                n=timesteps,
                                demand_lb=demand_lb,
                                demand_ub=demand_ub,
                                demand_dist=demand_dist,
                                leadtime_lb=leadtime_lb,
                                leadtime_ub=leadtime_ub,
                                leadtime_dist=leadtime_dist,
                                no_actions=no_actions,
                                no_states=no_states,
                                coded=False, 
                                fix=True,
                                ipfix=True,
                                method='DRL')


# PPO settings
# For coded inventory position 100.000 runs are recommonded
# For uncoded inventory position, ik denk wel meer dan 270 duizend

# Entropy blijft nog wel dalen
# Max is 270.000 tot nu toe

# nr of iterations for which a new sample is run -> 100000 is 2 uur ongeveer
ppo_iterations = 100000
# time steps to fill the buffer, also used to scale the rewards.
# Divide all rewards by ppo_iterations
timesteps_per_iteration = 35

# use a cooldown period for the buffer (i.e. run extra time steps)
cooldown_buffer = False
# clipping value for policy loss
ppo_epsilon = 0.2
# policy network learning rate
pi_lr = 1e-4
# value network learning rate
vf_lr = 1e-4
# nr of iterations after which the network weights are saved
ppo_save_freq = 1000
# nr of iterations over the entire batc
ppo_epochs = 10
# size of the mini batc
ppo_batch_size = 35
# number of hidden layers and nodes per layer
hidden_sizes = (64, 64)
# activation function of hidden layers
activation = nn.Tanh

for k in range(0, 6):
    print("Replication " + str(k))
    # set random seed
    RANDOM_SEED = k
    env = BeerGame(timesteps_per_iteration)
    set_seeds(env.env, RANDOM_SEED)
    ac = MLPActorCritic(env.env.observation_space, env.env.action_space,
                        hidden_sizes, activation=activation)

    experiment_name = "RN{} False,True,True".format(k)

    ppo_learning(env.env, ppo_iterations, timesteps_per_iteration, cooldown_buffer,
                 ppo_epsilon, pi_lr, vf_lr, ppo_save_freq, ppo_epochs,
                 ppo_batch_size, experiment_name)
    
# spinup.ppo_pytorch(env.env, actor_critic=ac,
#                     ac_kwargs={}, seed=0, steps_per_epoch=4000, epochs=50,
#                     gamma=0.99, clip_ratio=0.2, pi_lr=0.0003, vf_lr=0.001,
#                     train_pi_iters=80, train_v_iters=80, lam=0.97,
#                     max_ep_len=1000, target_kl=0.01, logger_kwargs={},
#                     save_freq=10)


