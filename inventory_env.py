"""@author: KevinG."""
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import gym
from gym import spaces

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

def generateLeadTimes(dist, n, lowerbound, upperbound):
    """
    Generate the leadtime using a predefined distribution.

    Returns: Integer
    """
    leadtimelist = []
    if dist == 'normal':
        for t in range(n):
            leadtimelist.append(random.randrange(lowerbound, upperbound + 1))
    elif dist == 'dataset1':
        leadtimelist = [0, 2, 0, 2, 4, 4, 4, 0, 2, 4, 1, 1, 0, 0, 1, 1, 0, 1,
                        1, 2, 1, 1, 1, 4, 2, 2, 1, 4, 3, 4, 1, 4, 0, 3, 3, 4]
    elif dist == 'dataset2':
        leadtimelist = [0,2,0,2,4,4,4,0,2,4,1,1,0,0,1,1,0,1,1,2,1,1,1,4,2,2,1,4,3,4,1,4,0,3,3,4]
    elif dist == 'dataset3':
        leadtimelist = [0,4,2,2,0,2,2,1,1,3,0,0,3,3,3,4,1,1,1,3,0,4,2,3,4,1,3,3,3,0,3,4,3,3,0,3]
    elif dist == 'dataset4':
        leadtimelist = [0,4,2,2,0,2,2,1,1,3,0,0,3,3,3,4,1,1,1,3,0,4,2,3,4,1,3,3,3,0,3,4,3,3,0,3]
    return leadtimelist

class InventoryEnv(gym.Env):
    """
    General Inventory Control Environment.

    Currently tested with:
    - A reinforcement learning model for supply chain ordering management:
      An application to the beer game - Chaharsooghi (2002)
    """

    def __init__(self, stockpoints_echelon, no_suppliers, no_customers,
                 no_stockpoints, connections,
                 unsatisfied_demand, goal, tsl, holding_costs, bo_costs,
                 initial_inventory, n, demand_lb, demand_ub, leadtime_lb,
                 leadtime_ub, demand_dist, leadtime_dist, no_actions,
                 no_states, coded, fix, ipfix, method):
        # Supply chain variables:
        self.stockpoints_echelon = stockpoints_echelon
        self.no_suppliers = no_suppliers
        self.no_customers = no_customers
        self.no_stockpoints = no_stockpoints
        self.connections = connections
        self.no_actions = no_actions
        self.no_states = no_states

        self.determine_potential_actions()
        self.determine_potential_states()

        # Total number of nodes
        self.no_nodes = sum(self.stockpoints_echelon)
        # Total number of echelons, including supplier and customer
        self.no_echelons = len(self.stockpoints_echelon)

        # Unsatisfied demand
        # This can be either 'backorders' or 'lost_sales'
        self.unsatisfied_demand = unsatisfied_demand

        # Goal of the method
        # This can be either 'target_service_level' or 'minimize_costs'
        self.goal = goal
        # The target service level, required if goal is 'target_service_level'
        self.tsl = tsl

        # Costs variables
        self.holding_costs = holding_costs
        self.bo_costs = bo_costs
        self.n = n
        self.initial_inventory = initial_inventory   # inventory level at start

        self.demand_lb = demand_lb
        self.demand_ub = demand_ub
        self.leadtime_lb = leadtime_lb
        self.leadtime_ub = leadtime_ub

        self.demand_dist = demand_dist
        self.leadtime_dist = leadtime_dist
        
        self.coded = coded
        self.fix = fix
        self.ipfix = ipfix
        self.method = method

    def determine_potential_actions(self):
        """
        Possible actions returned as Gym Space
        each period
        """
        self.action_space = spaces.Discrete(256)

    def determine_potential_states(self):
        """
        Based on the mean demand, we determine the maximum and minimum
        inventory to prevent the environment from reaching unlikely states
        """
        # Observation space consists of the current timestep and inventory positions of every echelon
        self.observation_space = spaces.Box(np.array([0,-30,-30,-30,-30]), 
                                            np.array([34,30,30,30,30]), 
                                            dtype=np.int32)
        
    def _generate_demand(self, t):
        """
        Generate the demand using a predefined distribution.

        Writes the demand to the orders table.
        """
        if self.demand_dist == 'poisson':
            demand_mean = random.randrange(self.demand_lb,
                                           self.demand_ub + 1)
            demand = np.random.poisson(demand_mean)
        elif self.demand_dist == 'normal':
            demand = random.randrange(self.demand_lb,
                                      self.demand_ub + 1)
        elif self.demand_dist == 'dataset1':
            demandlist = [15, 10, 8, 14, 9, 3, 13, 2, 13, 11, 3, 4, 6, 11, 15,
                          12, 15, 4, 12, 3, 13, 10, 15, 15, 3, 11, 1, 13, 10,
                          10, 0, 0, 8, 0, 14]
            demand = demandlist[t]
        elif self.demand_dist == 'dataset2':
            demandlist = [5,14,14,13,2,9,5,9,14,14,12,7,5,1,13,3,12,4,0,15,11,10,6,0,6,6,5,11,8,4,4,12,13,8,12]
            demand = demandlist[t]
        elif self.demand_dist == 'dataset3':
            demandlist = [15, 10, 8, 14, 9, 3, 13, 2, 13, 11, 3, 4, 6, 11, 15,
                          12, 15, 4, 12, 3, 13, 10, 15, 15, 3, 11, 1, 13, 10,
                          10, 0, 0, 8, 0, 14]
            demand = demandlist[t]
        elif self.demand_dist == 'dataset4':
            demandlist = [13,13,12,10,14,13,13,10,2,12,11,9,11,3,7,6,12,12,3,10,3,9,4,15,12,7,15,5,1,15,11,9,14,0,4]
            demand = demandlist[t]
        retailer_list, customer_list = np.nonzero(self.connections)
        for retailer, customer in zip(retailer_list[self.no_stockpoints:],
                                      customer_list[self.no_stockpoints:]):
            self.O[t, customer, retailer] = demand


    def reward(self, t):
        """
        Calculate the reward for the current period.

        In the case of minmize costs, the backorder and holding costs are
        calculated.
        When using target service level, the service level during lead time is
        calculated.

        Returns: Real
        """
        if self.goal == 'minimize_costs':
            backorder_costs = np.sum(self.BO[t] * self.bo_costs)
            holding_costs = np.sum(self.INV[t] * self.holding_costs)
            reward = backorder_costs + holding_costs
        elif self.goal == 'target_service_level':
            reward = self.tsl
            # HIER MOET LOGICA KOMEN ZODAT DE DEVIATE VAN DE TSL WORDT BEREKEND
        return reward

    def _initializeIP(self, t):
        """
        Initialize the inventory position for every node.

        Copies the inventory position from the previous timestep.
        """
        if t > 0:
            self.INV[t] = np.copy(self.INV[t-1])
            self.BO[t] = np.copy(self.BO[t-1])
            self.IP[t] = np.copy(self.IP[t-1])

    def _receiveIncomingDelivery(self, t):
        """
        Receives the incoming delivery for every stockpoint.

        Customers are not taken into account because of zero lead time
        Based on the amount stated in T
        """
        # TODO: Hier zou je ook kunnen loopen over nonzeros van T[t]
        # Loop over all suppliers and stockpoints
        for i in range(0, self.no_stockpoints + self.no_suppliers):
            # Loop over all stockpoints
            # Note that only forward delivery is possible, hence 'i +'
            for j in range(i + self.no_suppliers, self.no_stockpoints +
                           self.no_suppliers):
                delivery = self.T[t, i, j]
                self.INV[t, j] += delivery
                self.in_transit[t, i, j] -= delivery
                self.T[t, i, j] = 0

    # Demand is generated and orders upstream are fulfilled
    def _receiveIncomingOrders(self, t):
        i_list, j_list = np.nonzero(self.O[t])
        for i, j in zip(i_list, j_list):
            # Check if the current order can be fulfilled
            if self.INV[t, j] >= self.O[t, i, j]:
                self._fulfillOrder(t, j, i, self.O[t, i, j])
                # Else, fulfill as far as possible
            else:
                inventory = max(self.INV[t, j], 0)
                quantity = self.O[t, i, j] - inventory
                self._fulfillOrder(t, j, i, inventory)
                # Add to backorder if applicable
                if self.unsatisfied_demand == 'backorders':
                    self.BO[t, i, j] += quantity
        if self.unsatisfied_demand == 'backorders':
            i_list, j_list = np.nonzero(self.BO[t])
            for i, j in zip(i_list, j_list):
                inventory = self.INV[t, j]
                # If there are any backorders, fulfill them afterwards
                if inventory > 0:
                    # If the inventory is larger than the backorder
                    # Fulfill the whole backorder
                    backorder = self.BO[t, i, j]
                    if inventory >= backorder:
                        # Dit vind ik heel onlogisch, maar voorzover ik nu kan zien
                        # in de IPs komt de backorder nooit aan.
                        # Nu wel gedaan dmv fix
                        if self.fix:
                            self._fulfillOrder(t, j, i, backorder)
                        else:
                            self.INV[t, j] -= backorder
                        self.BO[t, i, j] = 0
                    # Else, fulfill the entire inventory
                    else:
                        self._fulfillOrder(t, j, i, inventory)
                        self.BO[t, i, j] -= inventory

    def _fulfillOrder(self, t, source, destination, quantity):
        # Customers don't have any lead time.
        if destination >= self.no_nodes - self.no_customers:
            leadtime = 0
        else:
            leadtime = self.leadTimes[t]
        # The order is fulfilled immediately for the customer
        # or whenever the leadtime is 0
        # TODO: Leadtime per connection, daarmee kan je voor customers
        # altijd een leadtime 0 doen
        if leadtime == 0:
            # The new inventorylevel is increased with the shipped quantity
            self.INV[t, destination] += quantity
        else:
            # If the order is not fulfilled immediately, denote the time when
            # the order will be delivered. This can not be larger than the horizon
            if t+leadtime < self.n:
                self.T[t+leadtime, source, destination] += quantity
            for k in range(t, min(t+leadtime, self.n) + 1):
                self.in_transit[k, source, destination] += quantity
        self.INV[t, source] -= quantity

    def _placeOutgoingOrder(self, t, action):
        # Retrieve all nodes with a connection to each other
        i_list, j_list = np.nonzero(self.connections)
        # Loop over connections, excluding the customers
        incomingOrders = np.sum(self.O[t], 0)
        for i, j in zip(i_list[:self.no_stockpoints], j_list[:self.no_stockpoints]):
            self.O[t+1, j, i] += incomingOrders[j] + action[j-self.no_customers]

    def _codeState(self, t):
        bo_echelon = np.sum(self.BO[t], 0)
        if self.ipfix:
            in_transit_echelon = np.sum(self.in_transit[t], 0)
        CIP = np.zeros([self.no_stockpoints + 1], dtype=int)
        CIP[0] = t+1
        for i in range(self.no_suppliers, self.no_stockpoints +
                       self.no_suppliers):
            if self.ipfix:
                self.IP[t, i] = self.INV[t, i] - bo_echelon[i] + in_transit_echelon[i]
            else:
                self.IP[t, i] = self.INV[t, i] - bo_echelon[i]
            if self.coded:
                if self.IP[t, i] < -6:
                    CIP[i] = 1
                elif self.IP[t, i] < -3:
                    CIP[i] = 2
                elif self.IP[t, i] < 0:
                    CIP[i] = 3
                elif self.IP[t, i] < 3:
                    CIP[i] = 4
                elif self.IP[t, i] < 6:
                    CIP[i] = 5
                elif self.IP[t, i] < 10:
                    CIP[i] = 6
                elif self.IP[t, i] < 15:
                    CIP[i] = 7
                elif self.IP[t, i] < 20:
                    CIP[i] = 8
                else:
                    CIP[i] = 9
            else:
                CIP[i] = self.IP[t, i]
        if self.method == 'Q-learning':
            return CIP[1:]
        else:
            return CIP

    def step(self, t, action, visualize=False):
        """
        Execute one step in the RL method.

        input: actionlist, visualize
        """
        # previous orders are received:
        self._initializeIP(t)
        if visualize:
            self._visualize(t, action="0. IP")
        self._generate_demand(t)
        if t < (self.n - 1):
            self._receiveIncomingDelivery(t)
            if visualize:
                self._visualize(t, action="1. Delivery")
        self._receiveIncomingOrders(t)
        if visualize:
            self._visualize(t, action="2. Demand")
        action_d = decode_action(action)
        self._placeOutgoingOrder(t, action_d)
        CIP = self._codeState(t)
        reward = self.reward(t)
        return CIP, -reward, 0, True

    def reset(self):
        """
        Reset the simulation.

        Has to be executed at the beginning of every iteration.
        """
        # Amount of inventory of echelon s in time t
        self.INV = np.zeros([self.n+1, self.no_nodes], dtype=int)
        # Amount of backorders of echelon s+1 to echelon s in time t
        self.BO = np.zeros([self.n+1, self.no_nodes, self.no_nodes], dtype=int)
        # Inventory position (inventory-backorders) of echelon s in time t
        self.IP = np.copy(self.INV)
        # Set the initial inventory level as given for every stockpoint
        self.INV[0] = self.initial_inventory
        # Assume unlimited production capacity for the suppliers
        for j in range(self.no_suppliers):
            self.INV[0, j] = 10000
        # Number of items ordered from stockpoint s to stockpoint s
        self.O = np.zeros([self.n+1, self.no_nodes, self.no_nodes], dtype=int)
        # Number of items arriving in time t from stockpoint s to stockpoint s
        self.T = np.copy(self.O)
        # Number of items in transit from stockpoint s to stockpoint s
        self.in_transit = np.copy(self.O)
        
        self.leadTimes = generateLeadTimes(self.leadtime_dist, self.n, 
                                           self.leadtime_lb, self.leadtime_ub)
        
        #Custom BeerGame Settings. Kloppen deze?
        self.O[1, 1, 0] = 4
        self.O[1, 2, 1] = 4
        self.O[1, 3, 2] = 4
        self.in_transit[0,3,4] = 8
        self.in_transit[1,3,4] = 4
        self.in_transit[0,2,3] = 4
        self.in_transit[1,2,3] = 4
        self.in_transit[0,1,2] = 4
        self.in_transit[1,1,2] = 4
        self.in_transit[0,0,1] = 4
        self.in_transit[1,0,1] = 4
        self.T[0, 3, 4] = 4
        self.T[1, 3, 4] = 4
        self.T[1, 2, 3] = 4   
        self.T[1, 1, 2] = 4   
        self.T[1, 0, 1] = 4   
        self.state = self._codeState(0)
        if self.method != 'Q-learning':
            self.state[0] -= 1
        return self.state

    def customSettings(self, table_name, time, source, destination, value):
        """
        Add custom settings to the simulation.

        Can be used to add initial shipments and demand.
        """
        table = eval(table_name)
        table[time, source, destination] = value

    def _visualize(self, t, action):
        # Initialize graph
        Graph = nx.DiGraph()
        i = 0
        label = {}
        pos = {}
        pos_edges = {}
        # EVEN TIJDELIJK OMDAT IK NOG GEEN IDEE HEB HOE WEL:
        postionlist = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
        # Draw nodes
        # enumerate() ?
        # https://medium.com/better-programming/stop-using-range-in-your-python-for-loops-53c04593f936
        for echelon in range(self.no_echelons):
            for stockpoint in range(self.stockpoints_echelon[echelon]):
                Graph.add_node(i)
                label[i] = self.INV[t, i]
                pos[i] = [echelon, 0.1*postionlist[stockpoint]]
                i += 1
        nx.draw(Graph, pos, edgelist=[])
        nx.draw_networkx_labels(Graph, pos, labels=label)

        # Draw edges for 'order_size' (T)
        for source in range(self.no_nodes):
            for destination in range(self.no_nodes):
                in_transit = self.in_transit[t, source, destination]
                if self.T[t, source, destination] > 0:
                    Graph.add_edge(source, destination,
                                   order_size=self.T[t, source, destination],
                                   in_transit=in_transit)
                elif self.connections[source][destination] == 1:
                    Graph.add_edge(source, destination, order_size=0,
                                   in_transit=in_transit)
        for p in pos:
            echelon = pos[p][0]
            height = pos[p][1]
            pos_edges[p] = [echelon, height + 0.005]
        labels_order_size = nx.get_edge_attributes(Graph, 'order_size')
        nx.draw_networkx_edge_labels(Graph, pos_edges,
                                     edge_labels=labels_order_size)
        for p in pos:
            echelon = pos[p][0]
            height = pos[p][1]
            pos_edges[p] = [echelon + 0.2, height + 0.005]
        labels_in_transit = nx.get_edge_attributes(Graph, 'in_transit')
        nx.draw_networkx_edge_labels(Graph, pos_edges,
                                     edge_labels=labels_in_transit)

        # draw edges for 'in_order' (O)
        for source in range(self.no_nodes):
            for destination in range(self.no_nodes):
                backorder = self.BO[t, source, destination]
                if self.O[t, source, destination] > 0:
                    Graph.add_edge(source, destination,
                                   in_order=self.O[t, source, destination],
                                   backorder=backorder)
                elif self.connections[destination][source] == 1:
                    Graph.add_edge(source, destination, in_order=0, backorder=backorder)

        nx.draw_networkx_edges(Graph, pos)
        for p in pos:
            echelon = pos[p][0]
            height = pos[p][1]
            pos_edges[p] = [echelon, height - 0.005]
        labels_in_order = nx.get_edge_attributes(Graph, 'in_order')
        nx.draw_networkx_edge_labels(Graph, pos_edges, edge_labels=labels_in_order)

        for p in pos:
            echelon = pos[p][0]
            height = pos[p][1]
            pos_edges[p] = [echelon + 0.2, height -0.005]
        labels_backorder = nx.get_edge_attributes(Graph, 'backorder')
        nx.draw_networkx_edge_labels(Graph, pos_edges, edge_labels=labels_backorder)
        plt.suptitle('Time: ' + str(t) + ' Action: ' + action, fontsize=11)
        plt.savefig(str(t)+action+'.png')
        plt.show()
        Graph.clear()
