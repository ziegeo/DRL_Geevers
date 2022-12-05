"""@author: KevinG."""
import numpy as np
from inventory_env import InventoryEnv
from ppo.ppo_training_functions import ppo_learning
from ppo.ppo_support_functions import set_seeds
from cases import Divergent

experiment_name                 = 'Divergent/DivergentFalsetest'

case                            = Divergent()


ppo_simulation_length       = 75
# length of initial simulation that is discarded
ppo_warmup_period           = 25

policy_results_states           = [[0,12,12,12,12]]

for k in range(10):
    print("Replication " + str(k))
    # Initialize environment
    env = InventoryEnv(case, case.action_low, case.action_high,
                       case.action_min, case.action_max, case.state_low, case.state_high, 
                       'DRL', fix=True)
    run_name = "RN{}".format(k)

    # set random seed
    set_seeds(env, k)
    
    # call learning function
    ppo_learning(env, False, experiment_name, run_name, 
                 network_activation, network_size, network_bias_init, network_weights_init,
                 ppo_evaluation_steps, ppo_evaluation_threshold, 
                 ppo_iterations, ppo_buffer_length, ppo_gamma, ppo_lambda, cooldown_buffer, 
                 ppo_epsilon, pi_lr, vf_lr, ppo_save_freq, ppo_epochs, ppo_batch_size,
                 ppo_simulation_runs, ppo_simulation_length, ppo_warmup_period, policy_results_states)