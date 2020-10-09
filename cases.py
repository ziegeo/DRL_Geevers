import numpy as np

class BeerGame:
    """Based on the beer game by Chaharsooghi (2008)."""

    def __init__(self):
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
        self.connections                     = np.array([
                                                [0, 1, 0, 0, 0, 0],
                                                [0, 0, 1, 0, 0, 0],
                                                [0, 0, 0, 1, 0, 0],
                                                [0, 0, 0, 0, 1, 0],
                                                [0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 0, 0]
                                                ])
        self.unsatisfied_demand              = 'backorders'                      # Determines what happens with unsatisfied demand, can be either 'backorders' or 'lost_sales'
        self.goal                            = 'minimize_costs'                  # Goal of the method, can be either 'target_service_level' or 'minimize_costs'
        self.tsl                             = 0.95                              # Target service level, required if goal is 'target_service_level'
        self.holding_costs                   = [0, 1, 1, 1, 1, 0]                # Initial inventory per stockpoint
        self.bo_costs                        = [2, 2, 2, 2, 2, 2]                # Holding costs per stockpoint
        self.initial_inventory               = [100000, 12, 12, 12, 12, 0]       # Backorder costs per stockpoint
        self.demand_dist                     = 'uniform'                         # Demand distribution, can be either 'poisson' or 'uniform'
        self.demand_lb                       = 0                                 # Lower bound of the demand distribution
        self.demand_ub                       = 15                                # Upper bound of the demand distribution
        self.leadtime_dist                   = 'uniform'                         # Leadtime distribution, can only be 'uniform'
        self.leadtime_lb                     = 0                                 # Lower bound of the leadtime distribution
        self.leadtime_ub                     = 4                                 # Upper bound of the leadtime distribution
        self.order_policy                    = False                             # Predetermined order policy, can be either 'X' or 'X+Y'
        self.horizon                         = 35
        self.divide                          = False

class Divergent:
    """ Based on the paper of Kunnumkal"""

    def __init__(self):
        # Supply chain variables
        # Number of nodes per echelon, including suppliers and customers
        # The first element is the number of suppliers
        # The last element is the number of customers
        self.stockpoints_echelon             = [1, 1, 3, 3]
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
        self.connections                = np.array([                        # Connections between every stockpoint
                                            [0, 1, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 1, 1, 1, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 1, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 1, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 1],
                                            [0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0]
                                            ])
        self.unsatisfied_demand              = 'backorders'                      # Determines what happens with unsatisfied demand, can be either 'backorders' or 'lost_sales'
        self.goal                            = 'minimize_costs'                  # Goal of the method, can be either 'target_service_level' or 'minimize_costs'
        self.tsl                             = 0.95                              # Target service level, required if goal is 'target_service_level'
        self.initial_inventory               = [1000000, 0, 0, 0, 0, 0, 0, 0]    # Initial inventory per stockpoint
        self.holding_costs                   = [0, 0.6, 1, 1, 1, 0, 0, 0]        # Holding costs per stockpoint
        self.bo_costs                        = [0, 0, 19, 19, 19, 0, 0, 0]       # Backorder costs per stockpoint
        self.demand_dist                     = 'poisson'                         # Demand distribution, can be either 'poisson' or 'uniform'
        self.demand_lb                       = 5                                 # Lower bound of the demand distribution
        self.demand_ub                       = 15                                # Upper bound of the demand distribution
        self.leadtime_dist                   = 'uniform'                         # Leadtime distribution, can only be 'uniform'
        self.leadtime_lb                     = 1                                 # Lower bound of the leadtime distribution
        self.leadtime_ub                     = 1                                 # Upper bound of the leadtime distribution
        self.order_policy                    = 'X'                           # Predetermined order policy, can be either 'X','X+Y' or '(s,S)
        self.action_low                      = np.array([-1,-1,-1,-1])
        self.action_high                     = np.array([1,1,1,1])
        self.action_min                      = np.array([0,0,0,0])
        self.action_max                      = np.array([150,150,150,150])
        self.state_low                       = np.zeros([13])
        self.state_high                      = np.array([8000, 6000,             # Total inventory and total backorders
                                                    2000,2000,2000,2000,         # Inventory per stockpoint
                                                    2000,2000,2000,         # Backorders per stockpoint
                                                    250,250,250,250])       # transport
        self.horizon                         = 75
        self.divide                          = 1000

class General:
    """ Based on the case of the CardBoard Company """

    def __init__(self):
        # Supply chain variables
        # Number of nodes per echelon, including suppliers and customers
        # The first element is the number of suppliers
        # The last element is the number of customers
        self.stockpoints_echelon             = [4, 4, 5, 5]
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
        self.connections                     = np.array([
                                                #0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
                                                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 0
                                                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 1
                                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 2
                                                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 3
                                                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],   # 4
                                                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],   # 5
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],   # 6
                                                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],   # 7
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],   # 8
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],   # 9
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],   # 10
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],   # 11
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],   # 12
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 13
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 14
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 15
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 16
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]    # 17
                                                ])
        self.unsatisfied_demand              = 'backorders'                      # Determines what happens with unsatisfied demand, can be either 'backorders' or 'lost_sales'
        self.goal                            = 'minimize_costs'                  # Goal of the method, can be either 'target_service_level' or 'minimize_costs'
        self.tsl                             = 0.95                              # Target service level, required if goal is 'target_service_level'
        self.initial_inventory               = [100000, 100000, 100000, 100000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.holding_costs                   = [0, 0, 0, 0, 0.6, 0.6, 0.6, 0.6, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]        # Holding costs per stockpoint
        self.bo_costs                        = [0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0]       # Backorder costs per stockpoint
        self.demand_dist                     = 'poisson'                         # Demand distribution, can be either 'poisson' or 'uniform'
        self.demand_lb                       = 10                                 # Lower bound of the demand distribution
        self.demand_ub                       = 10                                # Upper bound of the demand distribution
        self.leadtime_dist                   = 'uniform'                         # Leadtime distribution, can only be 'uniform'
        self.leadtime_lb                     = 1                                 # Lower bound of the leadtime distribution
        self.leadtime_ub                     = 1                                 # Upper bound of the leadtime distribution
        self.order_policy                    = 'X'                               # Predetermined order policy, can be either 'X' or 'X+Y'
        self.horizon                         = 500