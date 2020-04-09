"""@author: KevinG."""
import ast
import random
import pandas as pd
import numpy as np
from inventory_env import InventoryEnv
import matplotlib.pyplot as plt
import itertools


class DivergentSupplyChain:
    """
    Met de volgende characteristics:
        
    """
    def __init__(self):
        # Time variables
        self.t = 0                   # Current time
        self.iteration = 0           # Current iteration
        self.max_iteration = 1       # Number of iterations
        self.n = 35                  # Time horizon

        # Q-Learning variables
        self.alpha = 0.17            # Learning rate for Q-Learning

        # Exploration variables
        self.exploitation = 0.02     # Starting percentage of exploitation
        self.exploitation_max = 0.90
        self.exploitation_min = 0.02
        self.exploitation_iteration_max = 0.98
        self.exploitation_delta = (self.exploitation_max -
                                   self.exploitation_min) / self.max_iteration
        self.exploitation_iteration_delta = (self.exploitation_iteration_max -
                                             self.exploitation) / self.n

        # Supply chain variables
        # Number of nodes per echelon, including suppliers and customers
        # The first element is the number of suppliers
        # The last element is the number of customers
        self.stockpoints_echelon = [1, 3, 3]
        # Number of suppliers
        self.no_suppliers = self.stockpoints_echelon[0]
        # Number of customers
        self.no_customers = self.stockpoints_echelon[-1]
        # Number of stockpoints
        self.no_stockpoints = sum(self.stockpoints_echelon) - \
            self.no_suppliers - self.no_customers
        # Total number of nodes
        self.no_nodes = sum(self.stockpoints_echelon)
        # Total number of echelons, including supplier and customer
        self.no_echelons = len(self.stockpoints_echelon)

        # Connections between every stockpoint
        self.connections = np.array([
            [0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            ])

        # Unsatisfied demand
        # This can be either 'backorders' or 'lost_sales'
        self.unsatisfied_demand = 'lost_sales'

        # Goal of the method
        # This can be either 'target_service_level' or 'minimize_costs'
        self.goal = 'target_service_level'
        # Target service level, required if goal is 'target_service_level'
        self.tsl = 0.95
        # Costs, required if goal is 'minimize_costs'
        self.holding_costs = 1
        self.bo_costs = 2

        self.order_policy = 'X+Y'

        # State-Action variables
        self.possible_states = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.possible_actions = [0, 1, 2, 3]
        self.no_actions = len(self.possible_actions)

        # State space
        states = []
        state_list = itertools.product(self.possible_states,
                                       repeat=self.no_stockpoints)
        for state in state_list:
            states.append(self.tuple_to_str(state))

        # Action space
        actions = []
        action_list = itertools.product(self.possible_actions,
                                        repeat=self.no_stockpoints)
        for action in action_list:
            actions.append(self.tuple_to_str(action))

        # Initialize Q values
        self.q_table = pd.DataFrame(0, index=actions, columns=states)

        # Initialize environment
        self.env = InventoryEnv(stockpoints_echelon=self.stockpoints_echelon,
                                no_suppliers=self.no_suppliers,
                                no_customers=self.no_customers,
                                no_stockpoints=self.no_stockpoints,
                                no_nodes=self.no_nodes,
                                no_echelons=self.no_echelons,
                                connections=self.connections,
                                unsatisfied_demand=self.unsatisfied_demand,
                                goal=self.goal,
                                tsl=self.tsl,
                                no_actions=self.no_actions,
                                holding_costs=self.holding_costs,
                                bo_costs=self.bo_costs,
                                initial_inventory=12,
                                n=self.n)

    def tuple_to_str(self, tup):
        """Convert a tuple to string."""
        return str(list(map(int, tup)))

    def state_to_str(self, state):
        """Convert a list to string."""
        return str(list(map(int, state.tolist())))

    def get_next_action(self, state):
        """Determine the next action to be taken.

        Based on the exploitation rate
        Returns a list of actions
        """
        if random.random() <= self.exploitation_iteration:
            return self.greedy_action(state)
        else:
            return self.random_action()

    def greedy_action(self, state):
        """Retrieve the best action for the current state.

        Picks the best action corresponding to the highest Q value
        Returns a list of actions
        """
        action = self.q_table[str(state)].idxmax()
        return ast.literal_eval(action)

    def random_action(self):
        """Generate a random set of actions."""
        action = []
        for Y in range(self.no_stockpoints):
            Y = random.choice(self.possible_actions)
            action.append(Y)
        return action

    def get_Q(self, state):
        """Retrieve the highest Q value fo the current state."""
        return self.q_table[state].max()

    def update(self, old_state, new_state, action, reward):
        """Update the Q table."""
        new_state_q_value = self.get_Q(new_state)
        self.q_table.loc[action, old_state] += self.alpha * \
            (-reward + new_state_q_value - self.q_table.loc[action, old_state])

    def iteration(self):
        min_reward = 10000
        totalrewardlist = []
        while self.iteration < self.max_iteration:
            self.exploitation_iteration = self.exploitation
            totalreward = 0
            self.env._reset()
            old_state = [7, 7, 7]
            while self.t < self.n:
                action = self.get_next_action(old_state)
                # Take action and calculate r(t+1)
                new_state, reward, vf = self.env._step(self.t, action)
                totalreward += reward
                # print(totalreward)
                new_state = self.state_to_str(new_state)
                old_state = str(old_state)
                action = str(action)
                self.update(old_state, new_state, action, reward)
                old_state = new_state
                self.exploitation_iteration += self.exploitation_iteration_delta
                self.t += 1
            totalrewardlist.append(int(totalreward))
            if totalreward < min_reward:
                min_reward = totalreward
            # This exploitation is increased with increasing iteration number linearly
            self.exploitation += self.exploitation_delta
            self.t = 0
            print(self.iteration/self.max_iteration)
            self.iteration += 1
        print('Lowest reward: ' + str(min_reward))
        self.perform_greedy_policy()
        self.visualizecosts(totalrewardlist)

    def perform_greedy_policy(self):
        totalreward = 0
        self.env._reset()
        old_state = [7, 7, 7]
        policy = []
        while self.t < self.n:
            action = self.greedy_action(old_state)
            policy.append(action)
            new_state, reward, vf = self.env._step(self.t, action, True)
            print(reward)
            totalreward += reward
            new_state = self.state_to_str(new_state)
            old_state = new_state
            self.t += 1
        print('Greedy Policy reward: ' + str(totalreward))
        self.env.close()
        return policy

    def visualizecosts(self, values):
        plt.xlabel("Time")
        plt.ylabel("Costs")
        plt.title("Costs over time")
        plt.plot(values)
        plt.legend()
        plt.show()


env = DivergentSupplyChain()
DivergentSupplyChain.iteration(env)