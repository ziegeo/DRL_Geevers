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
ppo_evaluation_steps        = 500
#number of consecutive evaluation iterations without improvement
ppo_evaluation_threshold    = 250
# maximum number of iterations in learning run
ppo_iterations              = 50000
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
#TODO: chatgpt zegt 3e-4. Dat eens proberen?
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