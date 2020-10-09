# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:12:13 2020

@author: LotteH

This file contains the PPO algorithm function. 

"""

# imports
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import time
from gym.spaces import Box, Discrete
import numpy as np
import os

# import functions from other files
from ppo.ppo_support_functions import scale_input, compute_entropy_threshold, evaluate_policy
from ppo.ppo_network_functions import MLPActorCritic
from ppo.ppo_logger_functions import Logger
from ppo.ppo_buffer_functions import Buffer
from ppo.ppo_tensorboard_functions import add_ap_figure_to_tensorboard, write_experiment_information

def ppo_learning(env, benchmark, experiment_name, run_name,
                 network_activation, network_size, network_bias_initialization, network_weights_initialization,
                 ppo_evaluation_steps, ppo_evaluation_threshold, 
                 ppo_iterations, ppo_buffer_length, ppo_gamma, ppo_lambda, cooldown_buffer, 
                 ppo_epsilon, pi_lr, vf_lr, ppo_save_freq, ppo_epochs, ppo_batch_size, 
                 ppo_simulation_runs, ppo_simulation_length, ppo_warmup_period, policy_results_states):
    '''
    env                             = instantiation of the environment object
    benchmark                       = instantiation of the benchmark object
    experiment_name                 = name of ppo experiment
    run_name                        = name of run in ppo experiment
    network_activation              = activation function used in the neural network. E.g. nn.Tanh
    network_size                    = tuple of sizes of hidden layers of networks. E.g. (64, 64)
    network_bias_initialization     = value to initialize bias of neural network. Float
    network_weights_initialization  = type of weights initialization. normal or uniform. String
    ppo_evaluation_steps            = number of iterations between evaluation. Int
    ppo_evaluation_threshold        = number of consecutive evaluation iterations without improvement. Int
    ppo_iterations                  = number of iterations to train the model. Int
    ppo_buffer_length               = length of buffer. Int
    ppo_gamma                       = discount factor used for the GAE. Float
    ppo_lambda                      = lambda value used for the GAE. Float
    cooldown_buffer                 = indicator of whether we should use a cooldown period in the buffer. Boolean
    ppo_epsilon                     = clipping value for updating. Float
    pi_lr                           = learning rate for policy network. Float
    vf_lr                           = learning rate for the value network. Float
    ppo_save_freq                   = after x iterations, save the model weights and histograms to tensorboard
    ppo_epochs                      = nr of epochs (i.e. repetitions of the buffer) used in updating the model weights. 
    ppo_batch_size                  = batch size used to split the buffer for updating the model weights
    ppo_simulation_runs             = number of simulation runs to compute benchmark and as stopping criterion
    ppo_simulation_length           = length of simulation to compute benchmark and as stopping criterion
    ppo_warmup_period               = initial part of simulation that is discarded
    policy_results_states           = sample states (x times the demand) to demonstrate the resulting policy
    '''
    def compute_loss_pi(data):
        ''' 
        This function computes the policy loss, which is the custom loss function as defined by Schulman(2015)
        Includes entropy loss.
        '''
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        # policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - ppo_epsilon, 1 + ppo_epsilon) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss_pi
    
    def compute_loss_v(data):
        ''' This function computes the value network loss, which is the mean squared error '''
        obs, ret = data['obs'], data['ret']
        predictions = ac.v(obs)
        observed = ret
        loss_v = torch.nn.functional.mse_loss(predictions, observed) #((predictions - observed) ** 2).mean()
        return loss_v
    
    def update():
        ''' This function updates the network weights'''
        buffer.get()
        for epoch in range(ppo_epochs):
            pi_running_loss, v_running_loss = 0, 0
            datalist = buffer.get_batches_per_epoch(ppo_batch_size)
            for training_batch in datalist:
                # updating policy network
                pi_optimizer.zero_grad()
                output_loss_pi = compute_loss_pi(training_batch)
                output_loss_pi.backward()
                pi_optimizer.step()
                pi_running_loss += output_loss_pi.item()
                
                # updating value network
                vf_optimizer.zero_grad()
                output_loss_v = compute_loss_v(training_batch)
                output_loss_v.backward()
                vf_optimizer.step()
                v_running_loss += output_loss_v.item()
            logger.store(LossPi = pi_running_loss, LossV = v_running_loss)    

    # create actor critic network, and initialize its optimizers
    if network_activation == 'tanh':
        network_activation = nn.Tanh
    elif network_activation == 'relu':
        network_activation = nn.ReLU
    else:
        raise NotImplementedError
    ac = MLPActorCritic(env.observation_space, env.action_space, env.feasible_actions, network_size, network_activation,
                        network_bias_initialization, network_weights_initialization)
    print('ac initialized')
    
    pi_optimizer = Adam(ac.pi.parameters(), lr = pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr = vf_lr)
    
    # instantiate buffer and logger, tensorboard writer
    if isinstance(env.action_space, Box):
        action_dim = env.action_space.shape
        action_dim_ent = action_dim[0]
    elif isinstance(env.action_space, Discrete):
        action_dim = env.action_space.n
        action_dim_ent = action_dim
    else:
        raise NotImplementedError

    buffer = Buffer(env.observation_space.shape[0], action_dim, 
                    ppo_buffer_length, ppo_gamma, ppo_lambda, cooldown = cooldown_buffer)
    parent_directory = os.path.realpath('..')
    tensorboard_dir = os.path.join(parent_directory, "tensorboard_results", experiment_name, run_name)
    logger = Logger(exp_name = experiment_name, run_name = run_name)
    writer = SummaryWriter(tensorboard_dir)
    
    # compute benchmark costs + entropy threshold
    #benchmark_name, benchmark_cost = determine_benchmark(benchmark, ppo_simulation_runs, ppo_simulation_length)
    benchmark_name, benchmark_cost = 'Paper', 69
    # entropy_threshold = compute_entropy_threshold(action_dim_ent, 0.2)
    no_improvement_count = 0
    best_evaluation_mean, best_evaluation_upperbound = float('-inf'), float('-inf')
    
    # add informative text to the tensorboard file
    write_experiment_information(writer, env, network_activation, network_size, network_bias_initialization, 
                                 network_weights_initialization, ppo_evaluation_steps, ppo_evaluation_threshold, 
                                 ppo_iterations, ppo_buffer_length, ppo_gamma, ppo_lambda, cooldown_buffer, 
                                 ppo_epsilon, pi_lr, vf_lr, ppo_save_freq, ppo_epochs, ppo_batch_size, 
                                 ppo_simulation_runs, ppo_simulation_length, ppo_warmup_period, policy_results_states,
                                 benchmark_name, benchmark_cost)
    
    start_time = time.time()
    
    # start iterations
    for iteration in range(ppo_iterations):
        if env.case.__class__.__name__ == "BeerGame":
            env.leadtime_dist, env.demand_dist = 'uniform', 'uniform'
        # reset buffer every iteration
        buffer.reset_buffer()
        o = env.reset()
        # fill up buffer
        for t in range(buffer.max_size):
            o = torch.as_tensor([scale_input(env, o)], dtype = torch.float32)
            a, v, logp, entropy, stddev = ac.step(o)
            next_o, r, d, info_dict = env.step(a[0])
            if t < ppo_buffer_length:         # only add if it is within the buffer size, not for cooldown
                if t == 0:
                    value_firststate = v
                    if isinstance(env.action_space, Box):
                        low = env.action_space.low
                        high = env.action_space.high
                        max = env.action_max
                        min = env.action_min
                        action_clip = np.clip(a[0], low, high)
                        for i in range(len(action_clip)):
                            if high[0] == 1:
                                action_clip[i] = ((action_clip[i] - low[i]) / (high[i]-low[i])) * ((max[i] - min[i])) + min[i]
                            else:
                                action_clip[i] = action_clip[i] * (max[i] / high[i])
                        firstaction = [round(num) for num in action_clip]
                    elif isinstance(env.action_space, Discrete):
                        firstaction = a
                logger.store(VVals = v, TimeStepReturn = r, Entropy=entropy, Stddev=stddev, HoldingCosts=info_dict['holding_costs'], BackorderCosts=info_dict['backorder_costs'])
            
            # save and log observation, action, reward (scaled by number of iterations), value, logprob, entropy
            buffer.store(o, a, r/ppo_buffer_length, v, logp, entropy)

            # if d:  
            #     next_o = env.reset()
            #     print(next_o)
            # update obs (very important!)
            o = next_o
            
            # determine if the buffer has filled up
            buffer_finished = t == buffer.max_size - 1
            
            # we need to get the final value estimate
            if buffer_finished or d:
                if buffer_finished:
                    _, v, _, _, _ = ac.step(torch.as_tensor([scale_input(env, o)], dtype = torch.float32))
                else:
                    v = 0
                buffer.finish_path(v)
                o = env.reset()
        
        # perform PPO update
        update()
        
        # buffer.get()
        # logger.store(ReturnBuffer = )
        # check if learning can be stopped
        if iteration % ppo_evaluation_steps == 0:
            print('evaluating policy at iteration {}'.format(iteration))
            mean, best_evaluation_mean, best_evaluation_upperbound, no_improvement_count = evaluate_policy(ac, env, 1,
                                                                                                     ppo_simulation_runs, 
                                                                                                     ppo_simulation_length,
                                                                                                     ppo_warmup_period, 
                                                                                                     no_improvement_count, 
                                                                                                     best_evaluation_mean,
                                                                                                     best_evaluation_upperbound)
            if env.case.__class__.__name__ == "BeerGame":
                mean2, _, _, _ = evaluate_policy(ac, env, 2, 
                                                ppo_simulation_runs, 
                                                ppo_simulation_length,
                                                ppo_warmup_period, 
                                                no_improvement_count, 
                                                best_evaluation_mean,
                                                best_evaluation_upperbound)
                mean3, _, _, _ = evaluate_policy(ac, env, 3, 
                                                ppo_simulation_runs, 
                                                ppo_simulation_length,
                                                ppo_warmup_period, 
                                                no_improvement_count, 
                                                best_evaluation_mean,
                                                best_evaluation_upperbound)
                mean4, _, _, _ = evaluate_policy(ac, env, 4, 
                                                ppo_simulation_runs, 
                                                ppo_simulation_length,
                                                ppo_warmup_period, 
                                                no_improvement_count, 
                                                best_evaluation_mean,
                                                best_evaluation_upperbound)                                                                                                                                                                                                                                                                                                              
        
        # log info about iteration: iteration number, reward per time step, v values, policy loss, value loss, 
        # entropy, time taken per iteration
        logger.log_iteration_tabular('Iteration', iteration)
        logger.log_iteration_tabular('BenchmarkName', benchmark_name)
        logger.log_iteration_tabular('BenchmarkCost', benchmark_cost)
        logger.log_iteration_tabular('CurrentBestSolution', best_evaluation_mean)
        logger.log_iteration_tabular('CurrentBestUpperBound', best_evaluation_upperbound)
        tstep_return_stats  = logger.log_iteration_tabular('TimeStepReturn', with_min_max=True)
        vvals_stats         = logger.log_iteration_tabular('VVals', with_min_max=True)
        losspi_stats        = logger.log_iteration_tabular('LossPi', with_min_max=True)
        lossv_stats         = logger.log_iteration_tabular('LossV', with_min_max=True)
        entropy_stats       = logger.log_iteration_tabular('Entropy', with_min_max = True)
        stddev_stats        = logger.log_iteration_tabular('Stddev', average_only = True)
        backorder_stats     = logger.log_iteration_tabular('BackorderCosts', with_min_max = True)
        # setup_stats         = logger.log_iteration_tabular('Setups', with_min_max = True)
        holding_stats     = logger.log_iteration_tabular('HoldingCosts', with_min_max = True)
        logger.log_iteration_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
        
        # add scalars to tensorboard
        writer.add_scalar('3Loss/policy', losspi_stats[0], iteration)
        writer.add_scalar('3Loss/value', lossv_stats[0], iteration)
        writer.add_scalar('1PPOLearning/AverageRewardPPO', tstep_return_stats[0], iteration)
        writer.add_scalar('3Loss/PredictedValue', vvals_stats[0], iteration)
        writer.add_scalar('1PPOLearning/Entropy', entropy_stats[0], iteration)
        writer.add_scalar('1PPOLearning/Stddev', stddev_stats[0][0], iteration)
        writer.add_scalar('1PPOLearning/Dataset1', mean, iteration)
        if env.case.__class__.__name__ == "BeerGame":
            writer.add_scalar('1PPOLearning/Dataset2', mean2, iteration)
            writer.add_scalar('1PPOLearning/Dataset3', mean3, iteration)
            writer.add_scalar('1PPOLearning/Dataset4', mean4, iteration)
        writer.add_scalar('2KPIs/ValueFirstState', value_firststate, iteration)
        if isinstance(env.action_space, Box):
            for i in range(env.action_space.shape[0]):
                writer.add_scalar('2KPIs/Action {}'.format(i), firstaction[i], iteration)
        writer.add_scalar('2KPIs/BackorderCosts', backorder_stats[0], iteration)
        writer.add_scalar('2KPIs/HoldingCosts', holding_stats[0], iteration)
        
        # print every x iterations the progress, write histograms to tensorboard, save model (don't overwrite)
        if ((iteration + 1) % ppo_save_freq == 0) or (iteration == ppo_iterations - 1) or (iteration == 0): 
            logger.save_state(ac, env, pi_optimizer, vf_optimizer, 'model save', iteration)
            print('progress: {} iterations done'.format(iteration + 1), '\t',
                  'total time busy: {} minutes'.format(round((time.time()-start_time)/60, 2)))
                # add histograms of weights and of inputs used in training
            for name, param in ac.pi.named_parameters():
                writer.add_histogram('PolicyNet/{}'.format(name), param, iteration)
                writer.add_histogram('PolicyNetGrad/{}'.format(name), param.grad, iteration)
            for name, param in ac.v.named_parameters():
                writer.add_histogram('ValueNet/{}'.format(name), param, iteration)
                writer.add_histogram('ValueNetGrad/{}'.format(name), param.grad, iteration)
            if env.state_low[0] == 0:
                writer.add_histogram('InputValues/TotalInv', buffer.obs_buf[:, 0], iteration)
                writer.add_histogram('InputValues/TotalBO', buffer.obs_buf[:, 1], iteration)
                writer.add_histogram('InputValues/Stockpoint 1', buffer.obs_buf[:, 2], iteration)
                writer.add_histogram('InputValues/Stockpoint 2', buffer.obs_buf[:, 3], iteration)
                if env.case.__class__.__name__ == "BeerGame":
                    writer.add_histogram('InputValues/Stockpoint 3', buffer.obs_buf[:, 4], iteration)
                    writer.add_histogram('InputValues/Stockpoint 4', buffer.obs_buf[:, 5], iteration)
                    writer.add_histogram('InputValues/Backorders 1', buffer.obs_buf[:, 6], iteration)
                    writer.add_histogram('InputValues/Backorders 2', buffer.obs_buf[:, 7], iteration)
                    writer.add_histogram('InputValues/Backorders 3', buffer.obs_buf[:, 8], iteration)
                    writer.add_histogram('InputValues/Backorders 4', buffer.obs_buf[:, 9], iteration)
                    writer.add_histogram('InputValues/PreviousDemand 1', buffer.obs_buf[:, 10], iteration)
                    writer.add_histogram('InputValues/PreviousDemand 2', buffer.obs_buf[:, 11], iteration)
                    writer.add_histogram('InputValues/PreviousDemand 3', buffer.obs_buf[:, 12], iteration)
                    writer.add_histogram('InputValues/PreviousDemand 4', buffer.obs_buf[:, 13], iteration)
                    writer.add_histogram('InputValues/InTransit1', buffer.obs_buf[:, 14], iteration)
                    writer.add_histogram('InputValues/InTransit2', buffer.obs_buf[:, 15], iteration)
                    writer.add_histogram('InputValues/InTransit3', buffer.obs_buf[:, 16], iteration)
                    writer.add_histogram('InputValues/InTransit4', buffer.obs_buf[:, 17], iteration)
                    writer.add_histogram('InputValues/Transport1', buffer.obs_buf[:, 18], iteration)
                    writer.add_histogram('InputValues/Transport2', buffer.obs_buf[:, 19], iteration)
                    writer.add_histogram('InputValues/Transport3', buffer.obs_buf[:, 20], iteration)
                    writer.add_histogram('InputValues/Transport4', buffer.obs_buf[:, 21], iteration)
                elif env.case.__class__.__name__ == "Divergent":
                    writer.add_histogram('InputValues/Stockpoint 3', buffer.obs_buf[:, 4], iteration)
                    writer.add_histogram('InputValues/Stockpoint 4', buffer.obs_buf[:, 5], iteration)
                    writer.add_histogram('InputValues/Backorders 2', buffer.obs_buf[:, 6], iteration)
                    writer.add_histogram('InputValues/Backorders 3', buffer.obs_buf[:, 7], iteration)
                    writer.add_histogram('InputValues/Backorders 4', buffer.obs_buf[:, 8], iteration)
                    writer.add_histogram('InputValues/PreviousDemand 1', buffer.obs_buf[:, 9], iteration)
                    writer.add_histogram('InputValues/PreviousDemand 2', buffer.obs_buf[:, 10], iteration)
                    writer.add_histogram('InputValues/PreviousDemand 3', buffer.obs_buf[:, 11], iteration)
                    writer.add_histogram('InputValues/PreviousDemand 4', buffer.obs_buf[:, 12], iteration)
                    # writer.add_histogram('InputValues/InTransit1', buffer.obs_buf[:, 13], iteration)
                    # writer.add_histogram('InputValues/InTransit2', buffer.obs_buf[:, 14], iteration)
                    # writer.add_histogram('InputValues/InTransit3', buffer.obs_buf[:, 15], iteration)
                    # writer.add_histogram('InputValues/InTransit4', buffer.obs_buf[:, 16], iteration)
                    # writer.add_histogram('InputValues/Transport1', buffer.obs_buf[:, 17], iteration)
                    # writer.add_histogram('InputValues/Transport2', buffer.obs_buf[:, 18], iteration)
                    # writer.add_histogram('InputValues/Transport3', buffer.obs_buf[:, 19], iteration)
                    # writer.add_histogram('InputValues/Transport4', buffer.obs_buf[:, 20], iteration)
            else:
                if isinstance(env.action_space, Box):
                    for i in range(env.observation_space.shape[0]):
                        writer.add_histogram('InputValues/Stockpoint {}'.format(i), buffer.obs_buf[:, i], iteration)
            if isinstance(env.action_space, Box):
                for i in range(env.action_space.shape[0]):
                    writer.add_histogram('Exploration/Action {}'.format(i), buffer.act_buf[:, i], iteration)
            #add_ap_figure_to_tensorboard(writer, env, ac, policy_results_states, iteration)
        

        # if no_improvement_count >= ppo_evaluation_threshold and entropy_stats[0] < -2.5:
            # break
    
    # at the end of learning, visualize the resulting policy in tensorboard, and close the tensorboard and logger files
    print('learning finished at iteration {}'.format(iteration))
   # add_ap_figure_to_tensorboard(writer, env, ac, policy_results_states, iteration)
    logger.output_file.close()
    writer.close()
