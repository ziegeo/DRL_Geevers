"""@author: KevinG."""
import random
import time as ct
import numpy as np
from inventory_env import InventoryEnv
from cases.beergame import BeerGame

def encode_state(state):
    """Encode the state, so we can find it in the q_table."""
    encoded_state = (state[0] - 1) * 729
    encoded_state += (state[1] - 1) * 81
    encoded_state += (state[2] - 1) * 9
    encoded_state += (state[3] - 1)
    return int(encoded_state)

def decode_action(action):
    """Decode the action, so we can use it in the environment."""
    decoded_action = []
    decoded_action.append(int(action / 64))
    action = action % 64
    decoded_action.append(int(action / 16))
    action = action % 16
    decoded_action.append(int(action / 4))
    action = action % 4
    action = decoded_action.append(int(action))
    return decoded_action

def get_next_action_paper(time):
    """
    Determine the next action to take based on the policy from the paper.
    """
    actionlist = [106, 247, 35, 129, 33, 22, 148, 22, 231, 22, 22, 111, 111,
                  229, 192, 48, 87, 49, 0, 187, 254, 236, 94, 94, 89, 25, 22,
                  250, 45, 148, 106, 243, 151, 123, 67]
    return actionlist[time]

class QLearning:
    """Based on the beer game by Chaharsooghi (2008)."""

    def __init__(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.case                            = BeerGame()
        self.dist                            = 'uniform'

        # State-Action variables
        possible_states = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        possible_actions = [0, 1, 2, 3]

        self.no_states = len(possible_states) ** self.case.no_stockpoints
        self.no_actions = len(possible_actions) ** self.case.no_stockpoints

        self.initialize_q_learning()

        # Initialize environment
        self.env = InventoryEnv(case=self.case, action_low=0, action_high=0,
                                action_min=0, action_max=0, state_low=0, state_high=0,
                                actions=self.no_actions, coded=True, fix=FIX, ipfix=IPFIX,
                                method='Q-learning')

    def initialize_q_learning(self):
        # Time variables
        self.max_iteration      = 1000000     # max number of iterations (N)
        self.alpha              = 0.17        # Learning rate / Step size
        self.horizon            = 35          # Time steps
        self.stepsize           = 10000        # m volgens Mes? / Simulation frequency

        # Exploration Variables from paper
        # self.exploitation       = 0.02
        # exploitation_max        = 0.90
        # exploitation_min        = 0.02
        # self.exploitation_iter_max = 0.98
        # self.exploitation_delta = ((exploitation_max -
        #                             exploitation_min) /
        #                            (self.max_iteration - 1))

        # Epsilon-greedy policy
        self.exploitation           = 0.95

        # Initialize Q values with bad values
        self.initial_q_value = -7000
        # Without linear decrease in values:
        # self.q_table = np.full([self.horizon + 1, self.no_actions, 
                                  # self.no_states], self.initial_q_value)
        # With linear decrease in values:
        self.q_table = np.zeros([self.horizon + 1, self.no_actions, self.no_states])
        for i in range(self.horizon):
            self.q_table[i, :, :] = self.initial_q_value + 200 * i
        self.q_table[35, :, :] = 0

    def get_next_action(self, time, state, exploitation):
        """Determine the next action to be taken.

        Based on the exploitation rate
        Returns a list of actions
        """
        if random.random() <= exploitation:
            return self.greedy_action(time, state)
        return self.random_action()

    def greedy_action(self, time, state):
        """Retrieve the best action for the current state.

        Picks the best action corresponding to the highest Q value
        Returns a list of actions
        """
        return self.q_table[time][:, encode_state(state)].argmax()

    def random_action(self):
        """Generate a random set of actions."""
        return random.randint(0, self.no_actions - 1)

    def get_q(self, time, state_e):
        """Retrieve highest Q value of the state in the next time period."""
        return self.q_table[time + 1][:, state_e].max()

    def update(self, time, iteration, old_state, new_state, action, reward):
        """Update the Q table."""
        new_state_e = encode_state(new_state)
        old_state_e = encode_state(old_state)
        new_state_q_value = self.get_q(time, new_state_e)
        old_state_q_value = self.q_table[time][action, old_state_e]
        # If we do not want to take the initial value into account:
        # if new_state_q_value == self.initial_q_value + (270 * (time + 1)): new_state_q_value = 0
        a = 50000
        stepsize = max((a / (a + iteration)), 0.05)
        q_value = stepsize * (reward + new_state_q_value - old_state_q_value)
        # If we do not want to take the initial value into account:
        # if self.q_table[time][action, old_state_e] == self.initial_q_value + (270 * (time + 1)):
            # self.q_table[time][action, old_state_e] = q_value or reward
        # else:
        self.q_table[time][action, old_state_e] += q_value

    def iteration(self):
        """Iterate over the simulation."""
        q_valuelist, totalrewardlist1, totalrewardlist2, totalrewardlist3, totalrewardlist4 = [], [], [], [], []
        current_iteration, time = 0, 0
        while current_iteration < self.max_iteration:
            exploitation_iter = self.exploitation
            # exploitation_iter_delta = ((self.exploitation_iter_max -
            #                             self.exploitation) / (self.horizon - 1))
            self.case.leadtime_dist, self.case.demand_dist = self.dist, self.dist
            old_state = self.env.reset()
            while time < self.horizon:
                # action = get_next_action_paper(time)
                action = self.get_next_action(time, old_state, exploitation_iter)
                # Take action and calculate r(t+1)
                new_state, reward, _, _ = self.env.simulate(decode_action(action))
                self.update(time, current_iteration, old_state, new_state, action, reward)
                old_state = new_state
                # exploitation_iter += exploitation_iter_delta
                time += 1
            # self.exploitation += self.exploitation_delta
            # Simulation to show current performance
            if current_iteration % self.stepsize == 0:
                totalreward1, _, _ = self.perform_greedy_policy(False, 1)
                totalreward2, _, _ = self.perform_greedy_policy(False, 2)
                totalreward3, _, _ = self.perform_greedy_policy(False, 3)
                totalreward4, _, _ = self.perform_greedy_policy(False, 4)
                totalrewardlist1.append(int(totalreward1))
                totalrewardlist2.append(int(totalreward2))
                totalrewardlist3.append(int(totalreward3))
                totalrewardlist4.append(int(totalreward4))
                highest_q_start = self.q_table[0][:, 4920].max()
                q_valuelist.append(int(highest_q_start))
                print(highest_q_start)
                print('Current iteration: {}'.format(current_iteration))
                print('total time busy: {} minutes'.format(round((ct.time()-STARTTIME)/60, 2)))
            time = 0
            current_iteration += 1 
        totalreward1, latest_policy1, latest_rewardlist1 = self.perform_greedy_policy(False, 1)
        totalreward2, latest_policy2, latest_rewardlist2 = self.perform_greedy_policy(False, 2)
        totalreward3, latest_policy3, latest_rewardlist3 = self.perform_greedy_policy(False, 3)
        totalreward4, latest_policy4, latest_rewardlist4 = self.perform_greedy_policy(False, 4)        
        totalrewardlist1.append(int(totalreward1))
        totalrewardlist2.append(int(totalreward2))
        totalrewardlist3.append(int(totalreward3))
        totalrewardlist4.append(int(totalreward4))
        return totalrewardlist1, totalrewardlist2, totalrewardlist3, totalrewardlist4, q_valuelist, latest_policy1, latest_policy2, latest_policy3, latest_policy4, latest_rewardlist1, latest_rewardlist2, latest_rewardlist3, latest_rewardlist4

    def perform_greedy_policy(self, final, distributions):
        """
        Perform the greedy policy.

        Hence, it does not use a random action, but always choses the best
        action according to the Q_table
        """
        totalreward, time = 0, 0
        rewardlist, policy = [], []
        self.case.leadtime_dist, self.case.demand_dist = distributions, distributions
        old_state = self.env.reset()
        while time < self.horizon:
            # action = get_next_action_paper(time)
            action = self.greedy_action(time, old_state)
            policy.append(action)
            action_d = decode_action(action)
            new_state, reward, _, _ = self.env.simulate(action_d)
            rewardlist.append(reward)
            totalreward += reward
            old_state = new_state
            time += 1
        print('totalreward: {}'.format(totalreward))
        return totalreward, policy, rewardlist

for q in range(3):
    if q == 0:
        FIX = False
        IPFIX = False
    elif q == 1:
        FIX = True
        IPFIX = False
    else:
        FIX = True
        IPFIX = True
    print(str(FIX) + str(IPFIX))
    STARTTIME = ct.time()
    for k in range(0, 10):
        ENV = QLearning(k)
        print("Replication " + str(k))
        totalrewardlist1, totalrewardlist2, totalrewardlist3, totalrewardlist4, q_valuelist, latest_policy1, latest_policy2, latest_policy3, latest_policy4, latest_rewardlist1, latest_rewardlist2, latest_rewardlist3, latest_rewardlist4 = QLearning.iteration(ENV)
        file = open(str(FIX) + str(IPFIX) + "resultsNEW" + str(k) + ".txt", "w+")
        file.write("Total reward list (dataset1):")
        file.write(str(totalrewardlist1))
        file.write("\n Total reward list (dataset2):")
        file.write(str(totalrewardlist2))
        file.write("\n Total reward list (dataset3):")
        file.write(str(totalrewardlist3))
        file.write("\n Total reward list (dataset4):")
        file.write(str(totalrewardlist4))
        file.write("\n Highest Q-value for initial state:")
        file.write(str(q_valuelist))
        file.write("\n Q-values for initial state:")
        file.write(str(ENV.q_table[0][:, 4920]))  
        file.write("\n Policy (dataset1):")
        file.write(str(latest_policy1))
        file.write("\n Policy (dataset2):")
        file.write(str(latest_policy2))
        file.write("\n Policy (dataset3):")
        file.write(str(latest_policy3))
        file.write("\n Policy (dataset4):")
        file.write(str(latest_policy4))
        file.write("\n Reward list (dataset1):")
        file.write(str(latest_rewardlist1))
        file.write("\n Reward list (dataset2):")
        file.write(str(latest_rewardlist2))
        file.write("\n Reward list (dataset3):")
        file.write(str(latest_rewardlist3))
        file.write("\n Reward list (dataset4):")
        file.write(str(latest_rewardlist4))
        file.close()
