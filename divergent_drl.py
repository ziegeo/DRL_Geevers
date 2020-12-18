"""@author: KevinG."""
import numpy as np
from inventory_env import InventoryEnv
from ppo.ppo_training_functions import ppo_learning
from ppo.ppo_support_functions import set_seeds
from cases import Divergent

experiment_name                 = 'Divergent/DivergentFalsetest'

case                            = Divergent()

# PPO Settings
# activation function of network
network_activation          = 'tanh'
# size of network
network_size                = (64, 64)
# initial values of bias in network
network_bias_init           = 0.0            
# method of weight initialization for network (uniform or normal)
network_weights_init        = 'uniform'
# number of iterations between evaluation
ppo_evaluation_steps        = 100
#number of consecutive evaluation iterations without improvement
ppo_evaluation_threshold    = 250
# maximum number of iterations in learning run
ppo_iterations              = 30000
# length of one episode in buffer
ppo_buffer_length           = 256
# discount factor used in GAE calculations
ppo_gamma                   = 0.99
# lambda rate used in GAE calculations
ppo_lambda                  = 0.95
# indicator of using a cooldown period in the buffer (boolean)
cooldown_buffer             = False
# clipping value used in policy loss calculations
ppo_epsilon                 = 0.2
# learning rate for policy network
pi_lr                       = 1e-4
# learning rate for value network
vf_lr                       = 1e-4
# after x iterations, save  model weights and histograms to tensorboard
ppo_save_freq               = 500
# nr of epochs (i.e. repetitions of the buffer) used in updating the model weights
ppo_epochs                  = 10
# batch size used to split the buffer for updating the model weights
ppo_batch_size              = 64
# number of simulation runs to compute benchmark and as stopping criterion
ppo_simulation_runs         = 100
# length of simulation to compute benchmark and as stopping criterion
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