"""@author: KevinG."""
import random
import time as ct
import numpy as np
from inventory_env import InventoryEnv


def encode_state(state):
    """Encode the state, so we can find it in the q_table."""
    encoded_state = (state[0] - 1) * 729
    encoded_state += (state[1] - 1) * 81
    encoded_state += (state[2] - 1) * 9
    encoded_state += (state[3] - 1)
    return int(encoded_state)

def get_next_action_paper(time):
    """
    Determine the next action to take based on the policy from the paper.
    """
    actionlist = [106, 247, 35, 129, 33, 22, 148, 22, 231, 22, 22, 111, 111,
                  229, 192, 48, 87, 49, 0, 187, 254, 236, 94, 94, 89, 25, 22,
                  250, 45, 148, 106, 243, 151, 123, 67]
    return actionlist[time]

class BeerGame:
    """Based on the beer game by Chaharsooghi (2008)."""

    def __init__(self, seed):
        random.seed(seed)
        np.random.seed(seed)
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
        self.demand_dist = 'normal'
        demand_lb = 0
        demand_ub = 15
        self.leadtime_dist = 'normal'
        leadtime_lb = 0
        leadtime_ub = 4

        # State-Action variables
        possible_states = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        possible_actions = [0, 1, 2, 3]

        self.no_states = len(possible_states) ** no_stockpoints
        self.no_actions = len(possible_actions) ** no_stockpoints

        self.initialize_q_learning()
        
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
                                n=self.horizon,
                                demand_lb=demand_lb,
                                demand_ub=demand_ub,
                                demand_dist=self.demand_dist,
                                leadtime_lb=leadtime_lb,
                                leadtime_ub=leadtime_ub,
                                leadtime_dist=self.leadtime_dist,
                                no_actions=self.no_actions,
                                no_states=self.no_states,
                                coded=True,
                                fix=True,
                                ipfix=False,
                                method='Q-learning')
    
    def initialize_q_learning(self):
        # Time variables
        self.max_iteration = 1000000     # max number of iterations
        self.alpha = 0.17               # Learning rate
        self.horizon = 35
        self.stepsize = 10000           # m volgens Mes

        # Exploration Variables
        self.exploitation = 0.02        # Starting exploitation rate
        exploitation_max = 0.90
        exploitation_min = 0.02
        self.exploitation_iter_max = 0.98
        self.exploitation_delta = ((exploitation_max -
                                    exploitation_min) /
                                   (self.max_iteration - 1))
        # Initialize Q values
        self.q_table = np.zeros([self.horizon + 1, self.no_actions, self.no_states])
        #self.rewardcount = np.zeros([self.no_actions, self.no_states])        

    def custom_settings(self):
        """
        Can change the starting situation.
        """
        self.env.customSettings('self.O', 1, 1, 0, 4)
        self.env.customSettings('self.O', 1, 2, 1, 4)
        self.env.customSettings('self.O', 1, 3, 2, 4)
        self.env.customSettings('self.in_transit', 0, 3, 4, 8)
        self.env.customSettings('self.in_transit', 1, 3, 4, 4)
        self.env.customSettings('self.in_transit', 0, 2, 3, 4)
        self.env.customSettings('self.in_transit', 1, 2, 3, 4)
        self.env.customSettings('self.in_transit', 0, 1, 2, 4)
        self.env.customSettings('self.in_transit', 1, 1, 2, 4)
        self.env.customSettings('self.in_transit', 0, 0, 1, 4)
        self.env.customSettings('self.in_transit', 1, 0, 1, 4)
        self.env.customSettings('self.T', 0, 3, 4, 4)
        self.env.customSettings('self.T', 1, 3, 4, 4)
        self.env.customSettings('self.T', 1, 2, 3, 4)
        self.env.customSettings('self.T', 1, 1, 2, 4)
        self.env.customSettings('self.T', 1, 0, 1, 4)

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
        state_e = encode_state(state)
        action = self.q_table[time][:, state_e].argmax()
        return action

    def random_action(self):
        """Generate a random set of actions."""
        action = random.randint(0, self.no_actions - 1)
        return action

    def get_q(self, time, state_e):
        """Retrieve highest Q value of the state in the next time period."""
        return self.q_table[time + 1][:, state_e].max()

    def update(self, time, old_state, new_state, action, reward):
        """Update the Q table."""
        new_state_e = encode_state(new_state)
        old_state_e = encode_state(old_state)
        new_state_q_value = self.get_q(time, new_state_e)
        self.q_table[time][action, old_state_e] += self.alpha * \
            (reward + new_state_q_value -
             self.q_table[time][action, old_state_e])
        # Next part is for debugging, to see what states are visited often
        # And to see what the expected reward is
        #if time == 34:
        #    self.rewardcount[action, old_state_e] += 1

    def iteration(self):
        """Iterate over the simulation."""
        q_valuelist, totalrewardlist1, totalrewardlist2, totalrewardlist3, totalrewardlist4 = [], [], [], [], []
        current_iteration, time = 0, 0
        while current_iteration < self.max_iteration:
            exploitation_iter = self.exploitation
            exploitation_iter_delta = ((self.exploitation_iter_max -
                                        self.exploitation) / (self.horizon - 1))
            self.env.leadtime_dist, self.env.demand_dist = self.leadtime_dist, self.demand_dist
            old_state = self.env.reset()
            # self.custom_settings()
            while time < self.horizon:
                action = self.get_next_action(time, old_state, exploitation_iter)
                # Take action and calculate r(t+1)
                new_state, reward, _, _ = self.env.step(time, action)
                self.update(time, old_state, new_state, action, reward)
                old_state = new_state
                exploitation_iter += exploitation_iter_delta
                time += 1
            self.exploitation += self.exploitation_delta
            # Simulation to show current performance
            if current_iteration % self.stepsize == 0:
                totalreward1, _, _ = self.perform_greedy_policy(False, 'dataset1')
                totalreward2, _, _ = self.perform_greedy_policy(False, 'dataset2')
                totalreward3, _, _ = self.perform_greedy_policy(False, 'dataset3')
                totalreward4, _, _ = self.perform_greedy_policy(False, 'dataset4')
                totalrewardlist1.append(int(totalreward1))
                totalrewardlist2.append(int(totalreward2))
                totalrewardlist3.append(int(totalreward3))
                totalrewardlist4.append(int(totalreward4))
                highest_q = self.q_table[0][:, 4920].max()
                q_valuelist.append(int(highest_q))
                print(current_iteration)
                print('total time busy: {} minutes'.format(round((ct.time()-STARTTIME)/60, 2)))
            time = 0
            current_iteration += 1
        totalreward1, latest_policy1, latest_rewardlist1 = self.perform_greedy_policy(False, 'dataset1')
        totalreward2, latest_policy2, latest_rewardlist2 = self.perform_greedy_policy(False, 'dataset2')
        totalreward3, latest_policy3, latest_rewardlist3 = self.perform_greedy_policy(False, 'dataset3')
        totalreward4, latest_policy4, latest_rewardlist4 = self.perform_greedy_policy(False, 'dataset4')        
        totalrewardlist1.append(int(totalreward1))
        totalrewardlist2.append(int(totalreward2))
        totalrewardlist3.append(int(totalreward3))
        totalrewardlist4.append(int(totalreward4))
        # np.savetxt('RewardsCount.csv', self.rewardcount, delimiter=';')
        return totalrewardlist1, totalrewardlist2, totalrewardlist3, totalrewardlist4, q_valuelist, latest_policy1, latest_policy2, latest_policy3, latest_policy4, latest_rewardlist1, latest_rewardlist2, latest_rewardlist3, latest_rewardlist4

    def perform_greedy_policy(self, final, distributions):
        """
        Perform the greedy policy.

        Hence, it does not use a random action, but always choses the best
        action according to the Q_table
        """
        totalreward, time = 0, 0
        rewardlist, policy = [], []
        self.env.leadtime_dist, self.env.demand_dist = distributions, distributions
        old_state = self.env.reset()
        # self.custom_settings()
        while time < self.horizon:
            # action = get_next_action_paper(time)
            action = self.greedy_action(time, old_state)
            policy.append(action)
            new_state, reward, _, _ = self.env.step(time, action, final)
            rewardlist.append(reward)
            totalreward += reward
            old_state = new_state
            time += 1
        return totalreward, policy, rewardlist

STARTTIME = ct.time()
for k in range(4, 10):
    ENV = BeerGame(k)
    print("Replication " + str(k))
    totalrewardlist1, totalrewardlist2, totalrewardlist3, totalrewardlist4, q_valuelist, latest_policy1, latest_policy2, latest_policy3, latest_policy4, latest_rewardlist1, latest_rewardlist2, latest_rewardlist3, latest_rewardlist4 = BeerGame.iteration(ENV)
    file = open("resultsNEW" + str(k) + ".txt", "w+")
    file.write("Total reward list (dataset1):")
    file.write(str(totalrewardlist1))
    file.write("Total reward list (dataset2):")
    file.write(str(totalrewardlist2))
    file.write("Total reward list (dataset3):")
    file.write(str(totalrewardlist3))
    file.write("Total reward list (dataset4):")
    file.write(str(totalrewardlist4))
    file.write("Highest Q-value for initial state:")
    file.write(str(q_valuelist))   
    file.write("Policy (dataset1):")
    file.write(str(latest_policy1))
    file.write("Policy (dataset2):")
    file.write(str(latest_policy2))
    file.write("Policy (dataset3):")
    file.write(str(latest_policy3))
    file.write("Policy (dataset4):")
    file.write(str(latest_policy4))
    file.write("Reward list (dataset1):")
    file.write(str(latest_rewardlist1))
    file.write("Reward list (dataset2):")
    file.write(str(latest_rewardlist2))
    file.write("Reward list (dataset3):")
    file.write(str(latest_rewardlist3))
    file.write("Reward list (dataset4):")
    file.write(str(latest_rewardlist4))
    file.close()

