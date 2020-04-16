"""@author: KevinG."""
import random
import numpy as np
from inventory_env import InventoryEnv

class BeerGame:
    """
    Based on the beer game by Chaharsooghi (2008).
    """

    def __init__(self):
        # RANDOM NOG EVEN WAT AAN VERANDEREN
        random.seed(41)

        # Time variables
        self.t = 0                   # current time
        self.iteration = 0           # current iteration
        self.max_iteration = 1000     # max number of iterations
        self.alpha = 0.17            # Learning rate
        self.exploitation = 0.02     # Starting per
        self.n = 35

        # Exploration Variables
        self.exploitation_max = 0.90
        self.exploitation_min = 0.02
        self.exploitation_iteration_max = 0.98
        self.exploitation_delta = ((self.exploitation_max -
                                   self.exploitation_min) /
                                   (self.max_iteration - 1))

        # Supply chain variables
        # Number of nodes per echelon, including suppliers and customers
        # The first element is the number of suppliers
        # The last element is the number of customers
        self.stockpoints_echelon = [1, 1, 1, 1, 1, 1]
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
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0]
            ])

        # Unsatisfied demand
        # This can be either 'backorders' or 'lost_sales'
        self.unsatisfied_demand = 'backorders'

        # Goal of the method
        # This can be either 'target_service_level' or 'minimize_costs'
        self.goal = 'minimize_costs'
        # Target service level, required if goal is 'target_service_level'
        self.tsl = 0.95
        # Costs, required if goal is 'minimize_costs'
        self.holding_costs = 1
        self.bo_costs = 2

        self.order_policy = 'X+Y'

        # State-Action variables
        self.possible_states = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.possible_actions = [0, 1, 2, 3]

        self.no_states = len(self.possible_states) ** self.no_stockpoints
        self.no_actions = len(self.possible_actions) ** self.no_stockpoints

        # Initialize Q values
        self.q_table = np.zeros((self.no_actions, self.no_states))
        # self.q_table = pd.DataFrame(0, index=actions, columns=states)

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
                                holding_costs=self.holding_costs,
                                bo_costs=self.bo_costs,
                                initial_inventory=12,
                                n=self.n)

    def encode_state(self, state):
        """Encode the state, so we can find it in the q_table."""
        encoded_state = (state[0] - 1) * 729
        encoded_state += (state[1] - 1) * 81
        encoded_state += (state[2] - 1) * 9
        encoded_state += (state[3] - 1)
        return int(encoded_state)

    def encode_action(self, action):
        """Encode the action, so we can find it in the q_table."""
        encoded_action = action[0] * 64
        encoded_action += action[1] * 16
        encoded_action += action[2] * 4
        encoded_action += action[3]
        return int(encoded_action)

    def decode_action(self, action):
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
        state_e = self.encode_state(state)
        action = self.q_table[:, state_e].argmax()
        action_d = self.decode_action(action)
        return action_d

    def random_action(self):
        """Generate a random set of actions."""
        action = []
        for Y in range(self.no_stockpoints):
            Y = random.choice(self.possible_actions)
            action.append(Y)
        return action

    def get_Q(self, state_e):
        """Retrieve the highest Q value fo the current state."""
        return self.q_table[:, state_e].max()

    def update(self, old_state, new_state, action, reward):
        """Update the Q table."""
        action_e = self.encode_action(action)
        new_state_e = self.encode_state(new_state)
        old_state_e = self.encode_state(old_state)
        new_state_q_value = self.get_Q(new_state_e)
        self.q_table[action_e, old_state_e] += self.alpha * \
            (-reward + new_state_q_value - self.q_table[action_e, old_state_e])

    def iteration(self):
        totalrewardlist = []
        while self.iteration < self.max_iteration:
            self.exploitation_iteration = self.exploitation
            self.exploitation_iteration_delta = ((
                self.exploitation_iteration_max - self.exploitation) /
                (self.n - 1))
            totalreward = 0
            self.env._reset()
            old_state = [7, 7, 7, 7]
            while self.t < self.n:
                action = self.get_next_action(old_state)
                # Take action and calculate r(t+1)
                new_state, reward = self.env._step(self.t, action, False, 'learning')
                self.update(old_state, new_state, action, reward)
                old_state = new_state
                self.exploitation_iteration += self.exploitation_iteration_delta
                self.t += 1
            self.exploitation += self.exploitation_delta
            # Every 100 iterations, the greedy policy is performed to show the
            # current performance
            if self.iteration % 100 == 0:
                totalreward = self.perform_greedy_policy(False)
                totalrewardlist.append(int(totalreward))
                print(self.iteration)
            self.t = 0
            self.iteration += 1
        totalreward = self.perform_greedy_policy(True)
        totalrewardlist.append(int(totalreward))
        file = open("totalrewardlist.txt","w+")
        file.write(str(totalrewardlist))


    def perform_greedy_policy(self, final):
        self.t = 0
        totalreward = 0
        self.env._reset()
        old_state = [7, 7, 7, 7]
        policy = []
        while self.t < self.n:
            action = self.greedy_action(old_state)
            policy.append(action)
            if final:
                new_state, reward = self.env._step(self.t, action, False, 'greedy')
            else:
                new_state, reward = self.env._step(self.t, action, False, 'greedy')
            totalreward += reward
            old_state = new_state
            self.t += 1
        if final:
            print(policy)
        return totalreward

env = BeerGame()
BeerGame.iteration(env)