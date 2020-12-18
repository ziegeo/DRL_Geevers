# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:29:04 2020

@author: LotteH

This file contains functions for writing additional images or scalars to tensorboard.
"""

import matplotlib.pyplot as plt
import torch
import math
import numpy as np

from ppo.ppo_support_functions import scale_input

def write_experiment_information(writer, env, network_activation, network_size, 
                                 network_bias_initialization, network_weights_initialization, 
                                 ppo_evaluation_steps, ppo_evaluation_threshold, ppo_iterations, 
                                 ppo_buffer_length, ppo_gamma, ppo_lambda, cooldown_buffer, 
                                 ppo_epsilon, pi_lr, vf_lr, ppo_save_freq, ppo_epochs, 
                                 ppo_batch_size, ppo_simulation_runs, ppo_simulation_length, 
                                 ppo_warmup_period, policy_results_states, benchmark_name, 
                                 benchmark_cost):
    '''
    This function writes the information for the experiment to tensorboard, so you can view what 
    parameters are used:
        environment properties (demand mean, variance, capacity, production rate, setup time, 
                                inventory limits, initial inventory)
        environment costs (setup, holding, backorders)
        ppo settings (gamma, lambda, buffer size, epochs, batch size, epsilon, learning rates)
        ppo simulation criteria (nr and length of simulation, maximum nr of iterations)
        network properties (activation, size, initialization)
        actions (list of actions)
    '''
    #actions_list_string = env.actions_list
    #writer.add_text('Experiment information', 'Potential actions: total actions {}, list of actions {}'.format(env.action_space.shape[0],
    #                actions_list_string), 0)
    writer.add_text('Experiment information', 'Network properties: activation {}, network size {}, \
                    weight init {}, bias init {}'.format(network_activation, network_size, 
                    network_weights_initialization, network_bias_initialization), 0)
    writer.add_text('Experiment information', 'PPO simulation criteria: max_iterations {}, \
                    simulation runs {}, simulation length {}, warmup period {}, evaluation steps \
                    {}, consecutive periods without improvement {}'.format(ppo_iterations, 
                    ppo_simulation_runs, ppo_simulation_length, ppo_warmup_period, 
                    ppo_evaluation_steps, ppo_evaluation_threshold), 0)
    writer.add_text('Experiment information', 'PPO settings: gamma {}, lambda {}, buffer size {}, \
                    epochs {}, batch size {}, epsilon {}, policy learning rate {}, value learning \
                    rate {}'.format(ppo_gamma, ppo_lambda, ppo_buffer_length, ppo_epochs, 
                    ppo_batch_size, ppo_epsilon, pi_lr, vf_lr), 0)
    writer.add_text('Experiment information', 'Environment settings: state {}, action {}, action {}'.format(env.state_high, env.action_high, env.action_max), 0)
    # writer.add_text('Experiment information', 'Environment costs: holding costs {}, \
    #                 backorder costs {}'.format(env.holding_cost, env.backorder_cost), 0)
    # writer.add_text('Experiment information', 'Environment properties: products {}, demand {}, variance {}, capacity {}, \
    #                 production rate {}, setup time {}, inventory limit {}, initial inventory {}'.format(env.products, 
    #                 env.demand_mean, env.demand_variance, env.capacity, env.production_rate, env.setup_time, 
    #                 env.inventory_limit, env.initial_inventory), 0)
    # writer.add_text('Benchmark costs', 'Benchmark {}, costs = {}'.format(benchmark_name, round(benchmark_cost, ndigits = 2), 0))

def determine_figure_testcases(env, policy_result_states):
    '''
    This function creates three lists of states (no carryovers, carryovers of setup product 1, carryovers of setup
    product 2), as well as the x labels and y labels for the images to be created.
    Input: 
        environment
        list of how many times the demand mean is the inventory level (integers)
    '''
    # create inventory combinations
    # figures_observations = [[i * env.demand_mean[0], j * env.demand_mean[1]] 
    #                             for i in policy_result_states for j in policy_result_states]
    figures_observations = policy_result_states                               
    
    # add carryover nodes to state representation
    # figures_observations_nocarryover = policy_result_states
    
    # create figure labels
    x_labels_images = env.actions_list
    y_labels_images = [i for i in figures_observations]
    
    return figures_observations_nocarryover, x_labels_images, y_labels_images

def get_action_probabilities(observations, ac, env):
    ''' This function has as input the observations, and returns an array of action probabilities '''
    action_probs_observations = []
    for i in observations:
        o = torch.as_tensor(scale_input(env, i), dtype = torch.float32)
        action_probs_observations.append(ac.get_action_probs(o))
        
    return action_probs_observations

def properties_figure(action_probs, x_labels, y_labels):
    '''
    This function sets the properties of the figures:
        tick marks and labels
        color maps
        x and y labels
        title
    '''
    fig = plt.figure(figsize=(50,20))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(action_probs, cmap = 'Oranges', origin = 'upper')
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation = 30, horizontalalignment = 'center')
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, horizontalalignment = 'right')
    ax.set_ylabel('State')
    ax.set_xlabel('Action')
    fig.suptitle('Probability distribution over actions', fontsize = 16)
    return fig

def add_action_prob_figures(writer, action_probs, x_labels_images, y_labels_images, img_section, iteration):
    '''
    This function adds probability figures to tensorboard. It does so by splitting the state examples in chunks, for
    readability. However, if the action space is large, the figures are not readable.
    '''
    # determine the size of chunks
    # chunks = math.ceil(math.sqrt(len(y_labels_images)))
    # add figures to tensorboard
    # for i in range(chunks-1):
        # y_labels = y_labels_images[(i * chunks):min(len(y_labels_images),((i + 1) * chunks))]
        # action_probs_slice = action_probs[(i * chunks):min(len(y_labels_images),((i + 1) * chunks))]
        # writer.add_figure('{}/Set {}'.format(img_section, i + 1), 
        #                   properties_figure(action_probs_slice, x_labels_images, y_labels), iteration)
    writer.add_figure('{}/Set {}'.format(img_section, 1), 
                    properties_figure(action_probs, x_labels_images, y_labels_images), iteration)


def add_ap_figure_to_tensorboard(writer, env, ac, policy_result_states, iteration):
    '''
    This function writes action probability figures to tensorboard, to have insight in the resulting policy after
    ppo learning has finished.
    '''
    # Get necessary input states and image labels
    figures_observations, x_labels_images, y_labels_images = determine_figure_testcases(env, policy_result_states)
    
    # Get action probabilities
    action_probabilities = get_action_probabilities(figures_observations, ac, env)
    
    # Add figures
    add_action_prob_figures(writer, action_probabilities, x_labels_images, y_labels_images, 
                            'Action Probabilities', iteration)
