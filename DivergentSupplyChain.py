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


class DivergentSupplyChain:
    """
    Based on the paper of Kunnumkal shakalakadingdong.

    Divergent supply chain
    Poisson demand
    1 period lead-time

    """

    def __init__(self):
        # RANDOM NOG EVEN WAT AAN VERANDEREN
        self.seed = 1
        random.seed(self.seed)

        # Nieuw toegevoegd of aangepast:
        self.demand_lb = 5
        self.demand_ub = 15

        self.leadtime_lb = 0
        self.leadtime_ub = 0

        self.demand_dist = 'poisson'
        self.leadtime_dist = 'normal'
        # Holding costs per stockpoint
        self.holding_costs = [0, 0.6, 1, 1, 1, 0, 0, 0]
        self.bo_costs = [0, 0, 19, 19, 19, 0, 0, 0]
        self.n = 50
        self.initial_inventory = [1000, 45, 15, 15, 15, 0, 0, 0]

        # Time variables
        self.t = 0                      # current time
        self.current_iteration = 0      # current iteration
        self.max_iteration = 5000    # max number of iterations
        self.alpha = 0.17               # Learning rate

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
        self.stockpoints_echelon = [1, 1, 3, 3]
        # Number of suppliers
        self.no_suppliers = self.stockpoints_echelon[0]
        # Number of customers
        self.no_customers = self.stockpoints_echelon[-1]
        # Number of stockpoints
        self.no_stockpoints = sum(self.stockpoints_echelon) - \
            self.no_suppliers - self.no_customers

        # Connections between every stockpoint
        self.connections = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
            ])

        # Unsatisfied demand
        # This can be either 'backorders' or 'lost_sales'
        self.unsatisfied_demand = 'backorders'

        # Goal of the method
        # This can be either 'target_service_level' or 'minimize_costs'
        self.goal = 'minimize_costs'
        # Target service level, required if goal is 'target_service_level'
        self.tsl = 0.95

        self.order_policy = 'X+Y'

        # State-Action variables
        self.possible_states = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.possible_actions = [0, 1, 2, 3]

        self.no_states = len(self.possible_states) ** self.no_stockpoints
        self.no_actions = len(self.possible_actions) ** self.no_stockpoints

        # Initialize Q values
        self.q_table = np.zeros([self.n + 1, self.no_actions, self.no_states])
        self.rewardtable = np.zeros([self.no_actions, self.no_states])
        self.rewardcount = np.zeros([self.no_actions, self.no_states])

        # Initialize environment
        self.env = InventoryEnv(stockpoints_echelon=self.stockpoints_echelon,
                                no_suppliers=self.no_suppliers,
                                no_customers=self.no_customers,
                                no_stockpoints=self.no_stockpoints,
                                connections=self.connections,
                                unsatisfied_demand=self.unsatisfied_demand,
                                goal=self.goal,
                                tsl=self.tsl,
                                holding_costs=self.holding_costs,
                                bo_costs=self.bo_costs,
                                initial_inventory=self.initial_inventory,
                                n=self.n,
                                demand_ub=self.demand_ub,
                                demand_lb=self.demand_lb,
                                leadtime_lb=self.leadtime_lb,
                                leadtime_ub=self.leadtime_ub,
                                demand_dist=self.demand_dist,
                                leadtime_dist=self.leadtime_dist,
                                seed=self.seed)

    def _codeState(self, IP):
        # Coded IP in order to decrease the state space
        CIP = np.zeros([self.no_stockpoints], dtype=int)
        for i in range(self.no_suppliers, self.no_stockpoints +
                       self.no_suppliers):
            if IP[i] < -6:
                CIP[i-1] = 1
            elif IP[i] < -3:
                CIP[i-1] = 2
            elif IP[i] < 0:
                CIP[i-1] = 3
            elif IP[i] < 3:
                CIP[i-1] = 4
            elif IP[i] < 6:
                CIP[i-1] = 5
            elif IP[i] < 10:
                CIP[i-1] = 6
            elif IP[i] < 15:
                CIP[i-1] = 7
            elif IP[i] < 20:
                CIP[i-1] = 8
            else:
                CIP[i-1] = 9
        return CIP

    def get_next_action(self, state, exploitation):
        """Determine the next action to be taken.

        Based on the exploitation rate
        Returns a list of actions
        """
        if random.random() <= exploitation:
            return self.greedy_action(state)
        return self.random_action()

    def greedy_action(self, state):
        """Retrieve the best action for the current state.

        Picks the best action corresponding to the highest Q value
        Returns a list of actions
        """
        state_e = encode_state(state)
        action = self.q_table[self.t][:, state_e].argmax()
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
        """Retrieve highest Q value of the state in the next time period."""
        return self.q_table[self.t + 1][:, state_e].max()

    def update(self, old_state, new_state, action, reward):
        """Update the Q table."""
        action_e = encode_action(action)
        new_state_e = encode_state(new_state)
        old_state_e = encode_state(old_state)
        new_state_q_value = self.get_q(new_state_e)
        self.q_table[self.t][action_e, old_state_e] += self.alpha * \
            (-reward + new_state_q_value -
             self.q_table[self.t][action_e, old_state_e])

    def iteration(self):
        """Iterate over the simulation."""
        q_valuelist = []
        totalrewardlist = []
        while self.current_iteration < self.max_iteration:
            exploitation_iter = self.exploitation
            exploitation_iter_delta = ((self.exploitation_iter_max -
                                        self.exploitation) / (self.n - 1))
            totalreward = 0
            self.env.reset()
            old_state = [7, 7, 7, 7]
            while self.t < self.n:
                action = self.get_next_action(old_state, exploitation_iter)
                # Take action and calculate r(t+1)
                IP, reward = self.env.step(self.t, action)
                new_state = self._codeState(IP)
                self.update(old_state, new_state, action, reward)
                old_state = new_state
                exploitation_iter += exploitation_iter_delta
                self.t += 1
            self.exploitation += self.exploitation_delta
            # Every 1000 iterations, the greedy policy is performed to show the
            # current performance
            if self.current_iteration % 1000 == 0:
                totalreward, highest_q, _, _ = self.perform_greedy_policy(False)
                totalrewardlist.append(int(totalreward))
                q_valuelist.append(int(highest_q))
                print(totalreward)
                print(self.current_iteration/self.max_iteration)
            self.t = 0
            self.current_iteration += 1
        totalreward, _, policy, rewardlist = self.perform_greedy_policy(True)
        totalrewardlist.append(int(totalreward))
        np.savetxt('Q_table.csv', self.q_table[34], delimiter=';')
        np.savetxt('Rewards.csv', self.rewardtable, delimiter=';')
        np.savetxt('RewardsCount.csv', self.rewardcount, delimiter=';')
        file = open("results.txt", "w+")
        file.write("Total reward list:")
        file.write(str(totalrewardlist))
        file.write("Highest Q-value for initial state:")
        file.write(str(q_valuelist))
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
            IP, reward = self.env.step(self.t, action, False)
            new_state = self._codeState(IP)
            rewardlist.append(reward)
            totalreward += reward
            old_state = new_state
            self.t += 1
        highest_q = self.q_table[0][:, 4920].max()
        return totalreward, highest_q, policy, rewardlist


env = DivergentSupplyChain()
DivergentSupplyChain.iteration(env)
