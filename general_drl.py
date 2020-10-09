"""@author: KevinG."""
import numpy as np
from inventory_env import InventoryEnv
from ppo.ppo_training_functions import ppo_learning
from ppo.ppo_support_functions import set_seeds
from cases.general import General

experiment_name                 = 'CBCV300R1000NEW'

case = General()
horizon = 2
action_low                      = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
action_high                     = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
action_min                      = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
action_max                      = np.array([300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300])
state_low                       = np.zeros(47)
state_high                      = np.array([8000, 6000,             # Total inventory and total backorders
                                            2000,2000,2000,2000,2000,2000,2000,2000,2000,     # Inventory per stockpoint
                                            # 11, 12  ,13,  14,  15,  16,  17,  18,  19
                                            2000,2000,2000,2000,2000,2000,2000,2000,2000,     # Backorders per stockpoint
                                            #20, 21, 22, 23, 24, 25, 26, 27, 28
                                            250,250,250,250,250,250,250,250,250,       # Demand of previous period
                                            250,250,250,250,250,250,250,250,250,      # In transit
                                            250,250,250,250,250,250,250,250,250])       # transport


# PPO Settings
network_activation              = 'tanh'                        	# activation function of network
network_size                    = (64, 64, 64, 64)              	# size of network
network_bias_init               = 0.0                           	# initial values of bias in network
network_weights_init            = 'uniform'                     	# method of weight initialization for network (uniform or normal)
ppo_evaluation_steps            = 100                           	# number of iterations between evaluation
ppo_evaluation_threshold        = 250                           	# number of consecutive evaluation iterations without improvement
ppo_iterations                  = 50000                         	# maximum number of iterations in learning run
ppo_buffer_length               = 256                           	# length of one episode in buffer  #NOT SURE
ppo_gamma                       = 0.99                          	# discount factor used in GAE calculations
ppo_lambda                      = 0.95                          	# lambda rate used in GAE calculations
cooldown_buffer                 = False                         	# indicator of using a cooldown period in the buffer (boolean)
ppo_epsilon                     = 0.2                           	# clipping value used in policy loss calculations
pi_lr                           = 1e-4                          	# learning rate for policy network
vf_lr                           = 1e-4                          	# learning rate for value network
ppo_save_freq                   = 1000                          	# after x iterations, save  model weights and histograms to tensorboard
ppo_epochs                      = 10                            	# nr of epochs (i.e. repetitions of the buffer) used in updating the model weights
ppo_batch_size                  = 64                            	# batch size used to split the buffer for updating the model weights
ppo_simulation_runs             = 500                           	# number of simulation runs to compute benchmark and as stopping criterion
ppo_simulation_length           = 75                            	# length of simulation to compute benchmark and as stopping criterion
ppo_warmup_period               = 25                            	# length of initial simulation that is discarded
policy_results_states           = [[0,12,12,12,12]]
benchmark                       = False

for k in range(10):
    print("Replication " + str(k))
    # Initialize environment
    env = InventoryEnv(case, horizon, action_low, action_high,
                       action_min, action_max, state_low, state_high,
                       'DRL', "Continuous", fix=True)
    run_name = "RN{}".format(k)

    # set random seed
    set_seeds(env, k)
    
    # call learning function
    ppo_learning(env, benchmark, experiment_name, run_name, 
                 network_activation, network_size, network_bias_init, network_weights_init,
                 ppo_evaluation_steps, ppo_evaluation_threshold, 
                 ppo_iterations, ppo_buffer_length, ppo_gamma, ppo_lambda, cooldown_buffer, 
                 ppo_epsilon, pi_lr, vf_lr, ppo_save_freq, ppo_epochs, ppo_batch_size,
                 ppo_simulation_runs, ppo_simulation_length, ppo_warmup_period, policy_results_states)