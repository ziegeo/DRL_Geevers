# Resulting policy figures

from inventory_env import InventoryEnv
# from EnvironmentBenchmarks import BenchmarkCalculations
from ppo.ppo_buffer_functions import Buffer
from ppo.ppo_support_functions import scale_input, set_seeds
from ppo.ppo_network_functions import MLPActorCritic
from cases import BeerGame, Divergent
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats
import random

CASE        = "Divergent" 
ACTIES      = "Klein"
FIX         = True
NET         = (64, 64)
REWARD      = 1000
ppo_lambda  = 0.95
ppo_gamma   = 0.99
BEST        = True

def check_action_space(action):
    low = action_low
    high = action_high
    max = action_max
    min = action_min
    action_clip = np.clip(action, low, high)
    for i in range(len(action_clip)):
        if high[0] == 1:
            action_clip[i] = ((action_clip[i] - low[i]) / (high[i]-low[i])) * ((max[i] - min[i])) + min[i]
        else:
            action_clip[i] = action_clip[i] * (max[i] / high[i])
    action = [np.round(num) for num in action_clip]
    return action


if CASE == "BeerGame":
### BEERGAME NEURAL NET ####
    case            = BeerGame()
    state_low       = np.zeros([50])
    state_high      = np.array([4000,4000,1000,1000,1000,1000,1000,1000,1000,1000,30,30,30,30,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30])
    samplestate_bo      = np.array([0,4000,0,0,0,0,1000,1000,1000,1000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    samplestate_inv     = np.array([4000,0,1000,1000,1000,1000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    if ACTIES == "Klein":
        action_low      = np.array([0,0,0,0])
        action_high     = np.array([3,3,3,3])
        action_min      = np.array([0,0,0,0])
        action_max      = np.array([3,3,3,3])
        case.order_policy = "X+Y"
    if ACTIES == "Groot":
        action_low      = np.array([0,0,0,0])
        action_high     = np.array([5,5,5,5])
        action_min      = np.array([0,0,0,0])
        action_max      = np.array([30,30,30,30])
        case.order_policy = "X"
    case.divide = REWARD
    begin_sim = 0
    end_sim = 35
    if REWARD == 1000 and NET == (64, 64, 64, 64):
        DIRECTORY = 'results/DRL/' + CASE + '/' + ACTIES + '/' + str(FIX) + '/Reward1000/' 
    elif REWARD == 1000 and NET == (64, 64):
        DIRECTORY = 'results/DRL/' + CASE + '/' + ACTIES + '/' + str(FIX) + '/Reward100064/' 
    elif REWARD == 1 and NET == (64, 64):
        DIRECTORY = 'results/DRL/' + CASE + '/' + ACTIES + '/' + str(FIX) + '/64/' 
elif CASE == "Divergent":
    case = Divergent()
    case2 = Divergent()
    case2.order_policy = "BaseStock"
    action_low  = case.action_low 
    action_high = case.action_high
    action_min  = case.action_min 
    action_max  = case.action_max 
    state_low   = case.state_low
    state_high  = case.state_high
    begin_sim = 25
    end_sim = 75
    DIRECTORY = 'results/DRL/' + CASE + '/'
    REWARD = 1000
    case.divide = REWARD
    NET = (64, 64, 64, 64)
    samplestate_bo      = np.array([0, 6000,                # Total inventory and total backorders
                                    0,0,0,0,                # Inventory per stockpoint
                                    2000,2000,2000,         # Backorders per stockpoint
                                    0,0,0,0,                # Demand of previous period
                                    0,0,0,0,                # In transit
                                    0,0,0,0])
    samplestate_inv     = np.array([8000, 0,                # Total inventory and total backorders
                                    2000,2000,2000,2000,    # Inventory per stockpoint
                                    0,0,0,                  # Backorders per stockpoint
                                    250,250,250,250,        # Demand of previous period
                                    250,250,250,250,        # In transit
                                    250,250,250,250])

env = InventoryEnv(case=case,
                    action_low=action_low,
                    action_high=action_high,
                    action_min=action_min,
                    action_max=action_max,
                    state_low=state_low,
                    state_high=state_high,
                    actions="Continuous",
                    fix=FIX,
                    method='DRL')


env2 = InventoryEnv(case2, action_low, action_high, action_min, action_max,
                   state_low, state_high, 'Simulation', "Continuous", fix=FIX)
# b = BenchmarkCalculations(env)

# states, costs, production, demand_obs, backorder_obs, satisfied_demand_obs, theoretical_costs
# benchmark_results = b.run_soq(1, 1000)
# benchmark_inventory = np.array(benchmark_results[0])[1:, :]
# benchmark_backorders = np.array(benchmark_results[4])
# benchmark_production = np.array(benchmark_results[2])
# benchmark_costs = benchmark_results[1]

# fill buffer with final network

cooldown_buffer = False
directory = DIRECTORY

def get_ppo_policy_results(env, size, model):
    ac = MLPActorCritic(env.observation_space, env.action_space, env.feasible_actions, NET, nn.Tanh, 0.0, 'uniform')
    model = torch.load(directory + model)
    ac.load_state_dict(model)
    set_seeds(env, 0)

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
                    size, ppo_gamma, ppo_lambda, cooldown = cooldown_buffer)
    buffer.reset_buffer()
    o = env.reset()

    # fill up buffer
    for t in range(buffer.max_size):
        o = torch.as_tensor([scale_input(env, o)], dtype = torch.float32)
        a, v, logp, entropy, stdev = ac.step(o, True)
        next_o, r, d, info_dict = env.simulate(a[0])
        # save and log observation, action, reward (scaled by number of iterations), value, logprob, entropy
        buffer.store(o, a, r/size, v, logp, entropy)
        
        # update obs (very important!)
        o = next_o
        
        # determine if the buffer has filled up
        buffer_finished = t == buffer.max_size - 1
        
        # we need to get the final value estimate
        if buffer_finished:
            _, v, _, _,_ = ac.step(torch.as_tensor([scale_input(env, o)], dtype = torch.float32))
            buffer.finish_path(v)
            o = env.reset()
    buffer.get()
    state_norm = buffer.total_data['obs'].numpy()
    state = ((state_norm - -1) / (1+1)) * ((env.state_high - env.state_low)) + env.state_low
    state = np.rint(state)
    inventory = state[:, 2:6]
    if env.case.__class__.__name__ == "BeerGame":
        backorders = state[:, 6:10]
    elif env.case.__class__.__name__ == "Divergent":
        backorders = state[:, 6:9]
    else:
        raise NotImplemented
    o_bo = torch.as_tensor([scale_input(env, samplestate_bo)], dtype = torch.float32)
    actions_bo, _, _, _, stdev_bo = ac.step(o_bo)    
    actions_bo = check_action_space(actions_bo)   
    o_inv = torch.as_tensor([scale_input(env, samplestate_inv)], dtype = torch.float32)
    actions_inv, _, _, _, stdev_inv = ac.step(o_inv)
    actions_inv = check_action_space(actions_inv)    
    return inventory, backorders, buffer.total_data['rew'].numpy().mean(), actions_bo[0], actions_inv[0], stdev_bo[0], stdev_inv[0]

def get_benchmark_results(env, size):
    random.seed(0)
    np.random.seed(0)
    _ = env.reset()
    rewardlist, inventorylist, backorderlist = [], [], []
    for t in range(size):
            # _, reward, _, info = env.simulate([86, 42, 42, 42])
            _, reward, _, info = env.simulate([86, 23, 23, 23])
            inventorylist.append(info['holding_costs'])
            backorderlist.append(info['backorder_costs'])
            rewardlist.append(reward)
    return inventorylist, backorderlist, rewardlist


def plot_costs_over_time(prod1, prod2, costs, title, start, end):
    plt.figure(figsize = (8,5))
    plt.plot(prod1[start:end], label = 'Inventory', linewidth = 2)
    plt.plot(prod2[start:end], label = 'Backorders', linewidth = 2)
    plt.title(title + str(costs), fontsize = 18)
    plt.ylabel('inventory level')
    plt.ylim(bottom = 0, top = 500)
    plt.legend()
    fig_title = directory + title + str(end) + 'inv'
    plt.savefig(fig_title, dpi = 1000)

# Plot acties voor een given state als input
def plot_action_distribution(mean, std, x_min, x_max, title):
    fig, axs = plt.subplots(2, 2)
    k = 0
    for i in range(2):
        for j in range(2):
            x = np.linspace(x_min[k], x_max[k], (x_max[k]-x_min[k])*100)
            y = scipy.stats.norm.pdf(x,mean[k],std[k])
            axs[i, j].plot(x,y)
            plt.xlim(x_min[k],x_max[k])
            plt.ylim(0,1.5)
            axs[i, j].set_title('Action ' + str(k))
            k += 1
    fig_title = directory + title + 'dist'
    plt.savefig(fig_title, dpi = 1000)
    plt.clf()
        
    
# if BEST:
#     ppo_inventory, ppo_backorders, _, mean_bo, mean_inv, std_bo, std_inv = get_ppo_policy_results(env, 200, 'newresult1999.pt')
#     ppo_costs = np.sum(ppo_inventory * case.holding_costs[1:5], 1) + np.sum(ppo_backorders * case.bo_costs[2:5], 1) 
#     plot_costs_over_time(np.sum(ppo_inventory, 1), np.sum(ppo_backorders, 1), 
#                     np.sum(ppo_costs[begin_sim:end_sim]), 'Best PPO policy over time',begin_sim, end_sim)
#     plot_costs_over_time(np.sum(ppo_inventory, 1), np.sum(ppo_backorders, 1), 
#                     np.sum(ppo_costs[begin_sim:200]), 'Best PPO policy over time', begin_sim, 200)                
#     plot_action_distribution(mean_bo, std_bo, action_min, action_max, 'Best - BO')
#     plot_action_distribution(mean_inv, std_inv, action_min, action_max, 'Best - INV')

# ppo_inventory, ppo_backorders, _, mean_bo, mean_inv, std_bo, std_inv = get_ppo_policy_results(env, 200, 'newresult2999.pt')
# ppo_costs = np.sum(ppo_inventory * case.holding_costs[1:5], 1) + np.sum(ppo_backorders * case.bo_costs[2:5], 1) 
# print(ppo_costs)
# print(np.sum(ppo_costs[begin_sim:end_sim]))
# plot_costs_over_time(np.sum(ppo_inventory, 1), np.sum(ppo_backorders, 1), 
#                 np.sum(ppo_costs[begin_sim:end_sim]), 'TrainedNEW2 PPO policy over time', begin_sim, end_sim)
# plot_costs_over_time(np.sum(ppo_inventory, 1), np.sum(ppo_backorders, 1), 
#                 np.sum(ppo_costs[begin_sim:200]), 'TrainedBEW2 PPO policy over time', begin_sim, 200)

benchmark_inventory, benchmark_backorders, benchmark_reward = get_benchmark_results(env2, end_sim)
plot_costs_over_time(benchmark_inventory, benchmark_backorders, 
                np.sum(benchmark_reward), 'Benchmark policy over time', begin_sim, end_sim)
benchmark_inventory, benchmark_backorders, benchmark_reward = get_benchmark_results(env2, 200)
plot_costs_over_time(benchmark_inventory, benchmark_backorders, 
                np.sum(benchmark_reward), 'Benchmark policy over time', begin_sim, 200)


# plot_action_distribution(mean_bo, std_bo, action_min, action_max, 'Trained - BO')
# plot_action_distribution(mean_inv, std_inv, action_min, action_max, 'Trained - INV')
 
# Plot Regression Tree (SKlearn, Peternina & Das (2005))


# Plot Seaborn Plot
# Tabel met verschillende states, maar dan zou het een 50 bij 50 bij (dimensions van states) kunnen worden. Misschien groot, maar aan de andere kant valt het mss mee.

# Tabel met 
# WAT WIL IK ZIEN?
    # Verschillende states, waarbij we opzich kunnen definieren welke hoog en laag is, en dan daar steeds stappen tussen


def plot_actions_scatter():
    sns.scatterplot(x="")



