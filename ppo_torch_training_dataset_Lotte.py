# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:02:15 2020

@author: KevinG
"""


# imports
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import time
from torch.utils.data import Dataset, DataLoader

# import functions from other files
# from inventory_env import InventoryEnv
from support_functions_Lotte import set_seeds, scale_input
from AC_Lotte import MLPActorCritic
from ppo_logger_Lotte import Logger
from ppo_buffer_dataset_Lotte import Buffer
# from ppo_custom_tensorboard_Lotte import add_ap_figure_to_tensorboard, add_ap_scalars_to_tensorboard

''' This part is in the run_experiment_file
 environment settings + create environment
ENV_products                = 2                     # number of products in environment
ENV_capacity                = 6                     # hours of production
ENV_setup_time              = [0.0, 0.0]            # hours of time to set up production
ENV_demand_mean             = [5, 3]              # mean demand per product
ENV_demand_cov              = [1.0, 0.5]            # coefficient of variation of demand per product
ENV_setup_cost              = [0.5, 0.5]            # cost of setting up production per setup
ENV_holding_cost            = [1.0, 1.0]            # annual cost of holding inventory
ENV_nr_working_days         = 200.0                 # nr of working days (used for computing daily holding cost and
                                                    # annual demand rate)
ENV_service_level_target    = [0.98, 0.98]          # target SL for each product (used to compute the backorder cost)
ENV_nr_years                = 20                    # nr of years to compute average benchmark costs

env = LotSizingEnv(ENV_products,
                   ENV_capacity,
                   ENV_setup_time,
                   ENV_demand_mean,
                   ENV_demand_cov,
                   ENV_setup_cost,
                   ENV_holding_cost,
                   ENV_nr_working_days,
                   ENV_service_level_target)

# set random seed
RANDOM_SEED                 = 401                   # random seed to ensure reproducability
set_seeds(env, 401)

# PPO settings
ppo_iterations              = 200                    # nr of iterations for which a new sample is run
timesteps_per_iteration     = 256                    # time steps to fill the buffer, also used
                                                    # to scale the rewards. Divide all rewards by ppo_iterations
cooldown_buffer             = True                  # use a cooldown period for the buffer (i.e. run extra time steps)
ppo_epsilon                 = 0.2                   # clipping value for policy loss
pi_lr                       = 1e-4                  # policy network learning rate
vf_lr                       = 1e-4                  # value network learning rate
ppo_save_freq               = 100                   # nr of iterations after which the network weights are saved
ppo_epochs                  = 1                    # nr of iterations over the entire batch
ppo_batch_size              = 64                    # size of the mini batch

experiment_name = "test2"
'''

# support functions
def ppo_learning(env, ppo_iterations, timesteps_per_iteration, cooldown_buffer, ppo_epsilon, pi_lr, vf_lr,
                 ppo_save_freq, ppo_epochs, ppo_batch_size, experiment_name):
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - ppo_epsilon, 1 + ppo_epsilon) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss_pi

    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        predictions = ac.v(obs)
        observed = ret
        loss = ((predictions - observed) ** 2).mean()
        return loss

    def print_weight_bias_grad(msg):
        weight_idx = [0, 2, 4]
        print(msg)
        for i in weight_idx:
            print('layer prob net: ', i, '\n',
                  'weight: ', ac.pi.pi_net[i].weight, '\n',
                  'bias: ', ac.pi.pi_net[i].bias, '\n',
                  'grad weight: ', ac.pi.pi_net[i].weight.grad, '\n',
                  'grad bias: ', ac.pi.pi_net[i].bias.grad)
        for i in weight_idx:
            print('layer value net: ', i, '\n',
                  'weight: ', ac.v.v_net[i].weight, '\n',
                  'bias: ', ac.v.v_net[i].bias, '\n',
                  'grad weight: ', ac.v.v_net[i].weight.grad, '\n',
                  'grad bias: ', ac.v.v_net[i].bias.grad)

    def update():
        buffer.get()
        dataloader = DataLoader(buffer, batch_size = ppo_batch_size, shuffle = True)
        for epoch in range(ppo_epochs):
            pi_running_loss, v_running_loss = 0, 0
            for i, data in enumerate(dataloader):
                # updating policy network
                pi_optimizer.zero_grad()
                loss_pi = compute_loss_pi(data)
                loss_pi.backward()
                pi_optimizer.step()
                pi_running_loss += loss_pi.item()

                # updating value network
                vf_optimizer.zero_grad()
                loss_v = compute_loss_v(data)
                loss_v.backward()
                vf_optimizer.step()
                v_running_loss += loss_v.item()
            logger.store(LossPi = pi_running_loss, LossV = v_running_loss)
        return data

    def simulate_policy():
        print('simulating policy')
        total_return = 0
        o = env.reset()
        for timestep in range(timesteps_per_iteration):
            o = torch.as_tensor(scale_input(env, o), dtype = torch.float32)
            a, _, _, _ = ac.step(o)
            next_o, r, _, _ = env.step(timestep, a)
            total_return += r
        average_return = total_return / (timesteps_per_iteration)
        o = env.reset()
        return average_return

    # create actor critic network
    hidden_sizes = (64, 64)                 # number of hidden layers and nodes per layer
    activation = nn.Tanh                    # activation function of hidden layers
    ac = MLPActorCritic(env.observation_space,
                        env.action_space,
                        hidden_sizes,
                        activation = activation)                                      # create actor and critic net

    pi_optimizer = Adam(ac.pi.parameters(), lr = pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr = vf_lr)

    # instantiate buffer and logger, tensorboard writer
    buffer = Buffer(env.observation_space.shape[0],
                    env.action_space.n,
                    timesteps_per_iteration,
                    cooldown = cooldown_buffer)
    tensorboard_dir = "tensorboard/" + experiment_name
    logger = Logger(exp_name = experiment_name)
    writer = SummaryWriter(tensorboard_dir)

    # start run
    o, ep_ret, ep_len = env.reset(), 0, 0

    start_time = time.time()

    for iteration in range(ppo_iterations):
        buffer.reset_buffer()
        for t in range(buffer.max_size):
            o = torch.as_tensor(scale_input(env, o), dtype = torch.float32)
            a, v, logp, entropy = ac.step(o)
            next_o, r, d, _ = env.step(t, a)
            if t < timesteps_per_iteration:         # only add if it is within the buffer size, not for cooldown
                ep_ret += r/timesteps_per_iteration
                # ep_ret += r
                logger.store(VVals = v, Entropy = entropy)
            ep_len += 1

            # save and log
            buffer.store(o, a, r/timesteps_per_iteration, v, logp, entropy)
            # buffer.store(o, a, r, v, logp, entropy)

            # update obs (very important!)
            o = next_o

            epoch_ended = t == buffer.max_size - 1

            if epoch_ended:
                _, v, _, _ = ac.step(torch.as_tensor(o, dtype = torch.float32))
                buffer.finish_path(v)
                logger.store(EpRet = ep_ret)
                o, ep_ret, ep_len = env.reset(), 0, 0

        # perform PPO update
        update()

        # save model
        if (iteration % ppo_save_freq == 0) or (iteration == ppo_iterations - 1):
            logger.save_state(ac, pi_optimizer, vf_optimizer, 'model save', iteration)

        # log info about epoch
        logger.log_epoch_tabular('Iteration', iteration)
        epret_stats         = logger.log_epoch_tabular('EpRet', with_min_max=True)
        vvals_stats         = logger.log_epoch_tabular('VVals', with_min_max=True)
        losspi_stats        = logger.log_epoch_tabular('LossPi', with_min_max=True)
        lossv_stats         = logger.log_epoch_tabular('LossV', with_min_max=True)
        entropy_stats       = logger.log_epoch_tabular('Entropy', with_min_max = True)
        logger.log_epoch_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

        # add scalars to tensorboard
        writer.add_scalar('Loss/policy', losspi_stats[0], iteration)
        writer.add_scalar('Loss/value', lossv_stats[0], iteration)
        writer.add_scalars('Reward/Cumulative reward', {'ppo': epret_stats[0],
                                                        'benchmark': -69}, iteration)
        #TODO, BENCHMARK COSTS?
        writer.add_scalar('Reward/Predicted value', vvals_stats[0], iteration)
        writer.add_scalar('Agent behavior/Entropy', entropy_stats[0], iteration)

        # print every 100 iterations the progress
        if ((iteration + 1) % ppo_save_freq == 0):
            print('progress: {}%'.format((iteration+1) / ppo_iterations * 100), '\t',
                  'total time busy: {} minutes'.format(round((time.time()-start_time)/60, 2)))
                # add histograms of weights and of inputs used in training
            for name, param in ac.pi.named_parameters():
                writer.add_histogram('PolicyNet/{}'.format(name), param, iteration)
                writer.add_histogram('PolicyNetGrad/{}'.format(name), param.grad, iteration)
            for name, param in ac.v.named_parameters():
                writer.add_histogram('ValueNet/{}'.format(name), param, iteration)
                writer.add_histogram('ValueNetGrad/{}'.format(name), param.grad, iteration)
            writer.add_histogram('InputValues/Time', buffer.obs_buf[:, 0], iteration, 'stone')
            writer.add_histogram('InputValues/Stockpoint 1', buffer.obs_buf[:, 1], iteration, 'stone')
            writer.add_histogram('InputValues/Stockpoint 2', buffer.obs_buf[:, 2], iteration, 'stone')
            writer.add_histogram('InputValues/Stockpoint 3', buffer.obs_buf[:, 3], iteration, 'stone')
            writer.add_histogram('InputValues/Stockpoint 4', buffer.obs_buf[:, 4], iteration, 'stone')
            
 #           ADD ACTIONS 
            # Add figures to tensorboard to interpret the resulting policy
            # add_ap_scalars_to_tensorboard(writer, env, ac, iteration)

#        if ((iteration + 1) % 1000 == 0):
#            writer.add_scalar('Reward/Simulated Return', simulate_policy(), iteration)

    # add_ap_figure_to_tensorboard(writer, env, ac, iteration)
    logger.output_file.close()
    writer.close()
