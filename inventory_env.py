"""@author: KevinG."""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import gym
import random

# def _seed(self, seed=None):
#     self.np_random, seed = seeding.np_random(seed)
#     return [seed]


class InventoryEnv(gym.Env):
    """
    General Inventory Control Environment.

    Currently tested with:
    - A reinforcement learning model for supply chain ordering management:
      An application to the beer game - Chaharsooghi (2002)
    """

    def __init__(self, stockpoints_echelon, no_suppliers, no_customers,
                 no_stockpoints, no_nodes, no_echelons, connections,
                 unsatisfied_demand, goal, tsl, holding_costs, bo_costs,
                 initial_inventory, n):
        # Supply chain variables:
        self.stockpoints_echelon = stockpoints_echelon
        self.no_suppliers = no_suppliers
        self.no_customers = no_customers
        self.no_stockpoints = no_stockpoints
        self.no_nodes = no_nodes
        self.no_echelons = no_echelons
        self.connections = connections

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
        self.inventory_level = initial_inventory   # inventory level at start

    def _generateDemand(self, t):
        """
        Generate the demand using a predefined distribution.

        Writes the demand to the orders table.
        """
        if self.phase == 'learning':
            demand = random.randrange(0, 16)
        elif self.phase == 'greedy':
            demandlist = [15, 10, 8, 14, 9, 3, 13, 2, 13, 11, 3, 4, 6, 11, 15,
                          12, 15, 4, 12, 3, 13, 10, 15, 15, 3, 11, 1, 13, 10,
                          10, 0, 0, 8, 0, 14]
            demand = demandlist[t]

        retailer_list, customer_list = np.nonzero(self.connections)
        for retailer, customer in zip(retailer_list[self.no_stockpoints:],
                                      customer_list[self.no_stockpoints:]):
            self.O[t, customer, retailer] = demand

    def generateLeadtime(self, t, source, destination):
        """
        Generate the leadtime using a predefined distribution.

        Returns: Integer
        """
        # Customers don't have any lead time. Will be defined in connections
        # table in the future.
        if destination >= self.no_nodes - self.no_customers:
            leadtime = 0
        else:
            if self.phase == 'learning':
                leadtime = random.randrange(0, 5)
            elif self.phase == 'greedy':
                leadtimelist = [0, 2, 0, 2, 4, 4, 4, 0, 2, 4, 1, 1, 0, 0, 1, 1,
                                0, 1, 1, 2, 1, 1, 1, 4, 2, 2, 1, 4, 3, 4, 1, 4,
                                0, 3, 3, 4]
                leadtime = leadtimelist[t]
        return leadtime

# def transition(self, x, a, d):
#     m = self.max
#     return max(min(x + a, m) - d, 0)

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
            items_backorder = np.sum(self.BO[t])
            items_hold = np.sum(self.INV[t][1:-1])
            reward = items_backorder * self.bo_costs + \
                items_hold * self.holding_costs
        elif self.goal == 'target_service_level':
            reward = self.tsl
            # HIER MOET LOGICA KOMEN ZODAT DE DEVIATE VAN DE TSL WORDT BEREKEND
        return reward

    def valuefunction(self, t):
        """
        Calculate the value function for the future periods.

        Returns: Real
        """
        vf = 0
        for k in range(0, 35-t):
            for i in range(self.no_suppliers, self.no_stockpoints +
                           self.no_suppliers):
                items_hold = self.INV[t, i]
                items_backorder = self.BO[t, i-1, i]
                vf = vf + items_hold * self.holding_costs + \
                    items_backorder * self.bo_costs
        return vf

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
                        self.INV[t, j] -= backorder
                        self.BO[t, i, j] = 0
                    # Else, fulfill the entire inventory
                    else:
                        self._fulfillOrder(t, j, i, inventory)
                        self.BO[t, i, j] -= inventory

    def _fulfillOrder(self, t, source, destination, quantity):
        # Draw a random lead time
        leadtime = self.generateLeadtime(t, source, destination)
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
        # Misschien vervangen door -no_customers?
        # Loop over connections, excluding the customers
        for i, j in zip(i_list[:-1], j_list[:-1]):
            incomingOrders = np.sum(self.O[t], 0)
            self.O[t+1, j, i] += incomingOrders[j] + action[j-1]
            # self.O[t+1, j, i] += incomingOrders[j] + self.policy[t, j-1]

    # De volgende function kan beter in de case specific file, aangezien dit
    # natuurlijk heel erg afhangt van je eigen case
    def _codeState(self, t):
        BO_Echelon = np.sum(self.BO[t], 0)
        for i in range(self.no_suppliers, self.no_stockpoints +
                       self.no_suppliers):
            self.IP[t, i] = self.INV[t, i] - BO_Echelon[i]
            # DIT ZAL VAST MAKKELIJKER KUNNEN:
            if self.IP[t, i] < -6:
                self.CIP[t, i-1] = 1
            elif self.IP[t, i] < -3:
                self.CIP[t, i-1] = 2
            elif self.IP[t, i] < 0:
                self.CIP[t, i-1] = 3
            elif self.IP[t, i] < 3:
                self.CIP[t, i-1] = 4
            elif self.IP[t, i] < 6:
                self.CIP[t, i-1] = 5
            elif self.IP[t, i] < 10:
                self.CIP[t, i-1] = 6
            elif self.IP[t, i] < 15:
                self.CIP[t, i-1] = 7
            elif self.IP[t, i] < 20:
                self.CIP[t, i-1] = 8
            else:
                self.CIP[t, i-1] = 9

    def step(self, t, action, visualize=False, phase='learning'):
        self.phase = phase
        # previous orders are received:
        self._initializeIP(t)
        if visualize:
            self._visualize(t, action="0. IP")
        self._generateDemand(t)
        if t < (self.n - 1):
            self._receiveIncomingDelivery(t)
            if visualize:
                self._visualize(t, action="1. Delivery")
        self._receiveIncomingOrders(t)
        if visualize:
            self._visualize(t, action="2. Demand")
        self._placeOutgoingOrder(t, action)
        self._codeState(t)
        reward = self.reward(t)
        # valuefunction = self.valuefunction(t)
        return self.CIP[t], reward

    def reset(self):
        """
        Reset the simulation.

        Has to be executed at the beginning of every iteration.
        """
        # Amount of inventory of echelon s in time t
        self.INV = np.zeros([self.n+1, self.no_nodes])
        # Amount of backorders of echelon s+1 to echelon s in time t
        self.BO = np.zeros([self.n+1, self.no_nodes, self.no_nodes])
        # Inventory position (inventory-backorders) of echelon s in time t
        self.IP = np.copy(self.INV)
        # Coded IP in order to decrease the state space
        self.CIP = np.zeros([self.n+1, self.no_stockpoints])

        # Set the initial inventory level as given for every stockpoint
        for i in range(self.no_suppliers, self.no_stockpoints +
                       self.no_suppliers):
            self.INV[0, i] = self.inventory_level
        # Assume unlimited production capacity for the suppliers
        for j in range(self.no_suppliers):
            self.INV[0, j] = 10000

        # Number of items ordered from stockpoint s to stockpoint s
        self.O = np.zeros([self.n+1, self.no_nodes, self.no_nodes])
        # Number of items arriving in time t from stockpoint s to stockpoint s
        self.T = np.zeros([self.n+1, self.no_nodes, self.no_nodes])
        # Number of items in transit from stockpoint s to stockpoint s
        self.in_transit = np.copy(self.T)

    # def customSettings(self, table, time, source, destination):
    #     """
    #     Add custom settings to the simulation.

    #     Can be used to add initial shipments and demand.
    #     WORK IN PROGRESS
    #     """

        for s in range(1, 4):
            self.O[1, s, s-1] = 4
        # Set initial shipment as given BEERGAME SPECIFIC
        self.in_transit[0, 3, 4] = 8
        self.in_transit[1, 3, 4] = 4
        self.T[0, 3, 4] = 4
        self.T[1, 3, 4] = 4
        self.in_transit[0, 2, 3] = 4
        self.in_transit[1, 2, 3] = 4
        self.T[1, 2, 3] = 4
        self.in_transit[0, 1, 2] = 4
        self.in_transit[1, 1, 2] = 4
        self.T[1, 1, 2] = 4
        self.in_transit[0, 0, 1] = 4
        self.in_transit[1, 0, 1] = 4
        self.T[1, 0, 1] = 4

        # Set policy as given in the paper
        # self.policy = np.array([[1, 2, 2, 2],
        #                         [3, 3, 1, 3],
        #                         [0, 2, 0, 3],
        #                         [2, 0, 0, 1],
        #                         [0, 2, 0, 1],
        #                         [0, 1, 1, 2],
        #                         [2, 1, 1, 0],
        #                         [0, 1, 1, 2],
        #                         [3, 2, 1, 3],
        #                         [0, 1, 1, 2],
        #                         [0, 1, 1, 2],
        #                         [1, 2, 3, 3],
        #                         [1, 2, 3, 3],
        #                         [3, 2, 1, 1],
        #                         [3, 0, 0, 0],
        #                         [0, 3, 0, 0],
        #                         [1, 1, 1, 3],
        #                         [0, 3, 0, 1],
        #                         [0, 0, 0, 0],
        #                         [2, 3, 2, 3],
        #                         [3, 3, 3, 2],
        #                         [3, 2, 3, 0],
        #                         [1, 1, 3, 2],
        #                         [1, 1, 3, 2],
        #                         [1, 1, 2, 1],
        #                         [0, 1, 2, 1],
        #                         [0, 1, 1, 2],
        #                         [3, 3, 2, 2],
        #                         [0, 2, 3, 1],
        #                         [2, 1, 1, 0],
        #                         [1, 2, 2, 2],
        #                         [3, 3, 0, 3],
        #                         [2, 1, 1, 3],
        #                         [1, 3, 2, 3],
        #                         [1, 0, 0, 3]
        #                         ])
        
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
        for echelon in range(self.no_echelons):
            for stockpoint in range(self.stockpoints_echelon[echelon]):
                Graph.add_node(i)
                label[i] = int(self.INV[t, i])
                #label[i] = i
                pos[i] = [echelon, 0.1*postionlist[stockpoint]]
                i += 1
        nx.draw(Graph, pos, edgelist=[])
        nx.draw_networkx_labels(Graph, pos, labels=label)

        # Draw edges for 'order_size' (T)
        for source in range(self.no_nodes):
            for destination in range(self.no_nodes):
                in_transit = int(self.in_transit[t, source, destination])
                if self.T[t, source, destination] > 0:
                    Graph.add_edge(source, destination,
                                   order_size=int(self.T[t, source, destination]),
                                   in_transit=in_transit)
                elif self.connections[source][destination] == 1:
                    Graph.add_edge(source, destination, order_size=0, in_transit=in_transit)
        for p in pos:
            echelon = pos[p][0]
            height = pos[p][1]
            pos_edges[p] = [echelon, height + 0.005]
        labels_order_size = nx.get_edge_attributes(Graph, 'order_size')
        nx.draw_networkx_edge_labels(Graph, pos_edges, edge_labels=labels_order_size)
        for p in pos:
            echelon = pos[p][0]
            height = pos[p][1]
            pos_edges[p] = [echelon + 0.2, height + 0.005]
        labels_in_transit = nx.get_edge_attributes(Graph, 'in_transit')
        nx.draw_networkx_edge_labels(Graph, pos_edges, edge_labels=labels_in_transit)
        
        # draw edges for 'in_order' (O)
        for source in range(self.no_nodes):
            for destination in range(self.no_nodes):
                backorder = int(self.BO[t, source, destination])
                if self.O[t, source, destination] > 0:
                    Graph.add_edge(source, destination,
                                    in_order=int(self.O[t, source, destination]),
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
