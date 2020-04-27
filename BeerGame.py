"""@author: KevinG."""
import random
import numpy as np
from inventory_env import InventoryEnv


def encode_state(state):
    """Encode the state, so we can find it in the q_table."""
    encoded_state = (state[0] - 1) * 729
    encoded_state += (state[1] - 1) * 81
    encoded_state += (state[2] - 1) * 9
    encoded_state += (state[3] - 1)
    return int(encoded_state)


def encode_action(action):
    """Encode the action, so we can find it in the q_table."""
    encoded_action = action[0] * 64
    encoded_action += action[1] * 16
    encoded_action += action[2] * 4
    encoded_action += action[3]
    return int(encoded_action)


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


class BeerGame:
    """Based on the beer game by Chaharsooghi (2008)."""

    def __init__(self):
        # RANDOM NOG EVEN WAT AAN VERANDEREN
        random.seed(15)

        # Time variables
        self.t = 0                      # current time
        self.current_iteration = 0      # current iteration
        self.max_iteration = 1000000    # max number of iterations
        self.alpha = 0.17               # Learning rate
        self.n = 35

        # Exploration Variables
        self.exploitation = 0.02        # Starting exploitation rate
        self.exploitation_max = 0.90
        self.exploitation_min = 0.02
        self.exploitation_iter_max = 0.98
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

    def get_next_action(self, state):
        """Determine the next action to be taken.

        Based on the exploitation rate
        Returns a list of actions
        """
        if random.random() <= self.exploitation_iter:
            return self.greedy_action(state)
        return self.random_action()

    def greedy_action(self, state):
        """Retrieve the best action for the current state.

        Picks the best action corresponding to the highest Q value
        Returns a list of actions
        """
        state_e = encode_state(state)
        action = self.q_table[:, state_e].argmax()
        action_d = decode_action(action)
        return action_d

    def random_action(self):
        """Generate a random set of actions."""
        action = []
        for y in range(self.no_stockpoints):
            y = random.choice(self.possible_actions)
            action.append(y)
        return action

    def get_q(self, state_e):
        """Retrieve the highest Q value fo the current state."""
        return self.q_table[:, state_e].max()

    def update(self, old_state, new_state, action, reward):
        """Update the Q table."""
        action_e = encode_action(action)
        new_state_e = encode_state(new_state)
        old_state_e = encode_state(old_state)
        new_state_q_value = self.get_q(new_state_e)
        self.q_table[action_e, old_state_e] += self.alpha * \
            (-reward + new_state_q_value - self.q_table[action_e, old_state_e])

    def iteration(self):
        """Iterate over the simulation."""
        totalrewardlist = []
        while self.current_iteration < self.max_iteration:
            self.exploitation_iter = self.exploitation
            self.exploitation_iter_delta = ((
                self.exploitation_iter_max - self.exploitation) /
                                            (self.n - 1))
            totalreward = 0
            self.env.reset()
            old_state = [7, 7, 7, 7]
            while self.t < self.n:
                action = self.get_next_action(old_state)
                # Take action and calculate r(t+1)
                new_state, reward = self.env.step(self.t, action, False,
                                                  'learning')
                self.update(old_state, new_state, action, reward)
                old_state = new_state
                self.exploitation_iter += self.exploitation_iter_delta
                self.t += 1
            self.exploitation += self.exploitation_delta
            # Every 1000 iterations, the greedy policy is performed to show the
            # current performance
            if self.current_iteration % 1000 == 0:
                totalreward, _, _ = self.perform_greedy_policy(False)
                totalrewardlist.append(int(totalreward))
                print(self.current_iteration)
            self.t = 0
            self.current_iteration += 1
        totalreward, policy, rewardlist = self.perform_greedy_policy(True)
        totalrewardlist.append(int(totalreward))
        file = open("results.txt", "w+")
        file.write("Total reward list:")
        file.write(str(totalrewardlist))
        file.write("Q-Table:")
        file.write(str(self.q_table))
        file.write("Policy:")
        file.write(str(policy))
        file.write("Reward list:")
        file.write(str(rewardlist))

    def perform_greedy_policy(self, final):
        """
        Perform the greedy policy.

        Hence, it does not use a random action, but always choses the best
        action according to the Q_table
        """
        self.t = 0
        totalreward = 0
        rewardlist = []
        self.env.reset()
        old_state = [7, 7, 7, 7]
        policy = []
        while self.t < self.n:
            action = self.greedy_action(old_state)
            policy.append(action)
            # new_state, reward = self.env.step(self.t, action, final, 'greedy')
            new_state, reward = self.env.step(self.t, action, False, 'learning')
            rewardlist.append(reward)
            totalreward += reward
            old_state = new_state
            self.t += 1
        if self.current_iteration == self.max_iteration - 1000:
            file1 = open("previous_q_table.txt", "w+")
            file1.write(str(self.q_table))
            file1.close()
        return totalreward, policy, rewardlist


env = BeerGame()
BeerGame.iteration(env)
