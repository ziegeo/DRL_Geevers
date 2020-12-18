"""@author: KevinG."""
import time as ct
import numpy as np
import random
from inventory_env import InventoryEnv
from cases import BeerGame



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

def get_next_action(time, rnstream, dataset):
    """
    Determine the next action to take based on the policy from the paper.
    """
    if dataset == 1:
        actionlist = [106, 247, 35, 129, 33, 22, 148, 22, 231, 22, 22, 111, 111,
                  229, 192, 48, 87, 49, 0, 187, 254, 236, 94, 94, 89, 25, 22,
                  250, 45, 148, 106, 243, 151, 123, 67]
        if rnstream == 0:
            actionlist = [76,160,56,200,218,92,237,146,165,227,250,58,108,31,124,31,110,121,211,20,26,11,59,120,94,55,130,231,226,251,121,109,215,151,78]
        elif rnstream == 1:
            actionlist = [53,16,173,136,233,194,242,220,207,108,226,52,91,48,127,20,116,164,36,22,71,14,98,121,97,163,2,160,197,21,69,9,121,255,231]
        elif rnstream == 2:
            actionlist = [180,131,147,213,84,39,33,252,103,4,208,62,89,37,140,35,24,192,198,232,56,138,58,181,219,244,124,241,197,219,102,2,56,167,30]
        elif rnstream == 3:
            actionlist = [199,216,50,147,255,122,125,173,216,251,136,145,179,24,102,126,67,128,68,231,197,72,179,96,64,187,114,34,242,37,39,48,147,36,89]
        elif rnstream == 4:
            actionlist = [191,245,158,210,192,129,163,126,9,26,135,190,199,24,155,43,36,199,228,34,107,49,220,80,51,42,78,135,172,41,158,234,73,24,239]
        elif rnstream == 5:
            actionlist = [63,9,68,4,182,18,115,164,148,239,176,193,162,12,66,59,49,75,10,6,9,22,23,24,58,141,170,217,24,41,239,56,123,216,153]
        elif rnstream == 6:
            actionlist = [76,108,240,237,72,18,175,79,31,171,37,237,1,57,169,33,26,126,56,45,8,6,69,45,146,108,75,187,60,21,239,214,61,79,99]
        elif rnstream == 7:
            actionlist = [191,136,232,8,119,138,250,103,126,43,20,136,64,11,155,33,34,139,29,70,62,8,122,62,55,23,85,232,170,7,81,189,147,6,81]
        elif rnstream == 8:
            actionlist = [9,156,237,212,152,117,102,24,76,98,84,35,45,58,162,25,172,139,125,6,17,114,35,160,229,159,185,24,74,146,46,68,220,105,228]
        elif rnstream == 9:
            actionlist = [19,80,72,158,196,27,224,177,161,95,150,71,168,139,101,95,25,136,144,151,47,90,41,35,222,125,254,11,253,196,111,63,82,136,42]
    elif dataset == 2:
        if rnstream == 0:
            actionlist = [76,9,238,204,130,92,250,146,233,227,250,130,177,84,59,135,127,190,158,103,105,85,65,196,232,40,5,101,7,19,182,134,5,23,142]
        elif rnstream == 1:
            actionlist = [53,241,80,205,73,194,208,220,246,108,226,31,0,118,38,91,138,40,24,61,108,99,235,74,90,9,99,3,81,177,38,249,152,103,127]
        elif rnstream == 2:
            actionlist = [180,219,56,133,240,54,35,252,152,4,208,226,75,84,118,86,37,48,33,52,93,43,159,51,25,1,2,1,23,204,80,211,31,109,128]
        elif rnstream == 3:
            actionlist = [199,214,43,198,172,89,4,173,120,251,136,115,152,48,110,52,171,102,68,159,174,85,185,239,96,8,1,11,30,136,54,181,192,118,185]
        elif rnstream == 4:
            actionlist = [191,106,254,159,188,150,148,126,216,26,135,236,20,183,167,12,64,85,78,186,128,53,63,201,101,15,4,19,189,134,142,127,2,231,56]
        elif rnstream == 5:
            actionlist = [63,91,9,114,226,56,115,164,100,239,176,193,155,88,55,70,49,76,49,28,19,37,83,80,17,2,1,25,110,111,109,69,3,20,26]
        elif rnstream == 6:
            actionlist = [76,15,229,205,142,56,175,79,55,235,37,193,42,73,73,51,193,93,129,161,104,81,174,138,94,10,44,32,24,203,139,164,39,200,64]
        elif rnstream == 7:
            actionlist = [191,244,165,88,122,92,250,103,35,43,20,240,203,46,107,81,66,123,48,160,36,38,99,153,70,6,2,5,225,192,105,10,16,238,252]
        elif rnstream == 8:
            actionlist = [9,204,102,220,90,117,102,24,77,98,84,118,161,38,60,43,49,73,93,43,30,26,53,22,8,2,1,9,191,210,9,243,8,27,32]
        elif rnstream == 9:
            actionlist = [19,105,104,35,169,98,219,177,161,95,150,238,196,48,47,63,122,59,30,55,35,16,78,77,45,8,6,1,169,225,19,3,58,69,97]
    elif dataset == 3:
        if rnstream == 0:
            actionlist = [76,160,56,201,156,176,233,86,165,8,160,190,96,168,171,228,65,128,1,37,148,190,11,168,174,55,130,91,52,11,104,157,47,242,128]
        elif rnstream == 1:
            actionlist = [53,16,173,153,118,40,223,220,92,178,195,61,86,160,225,173,205,167,80,53,191,154,219,35,248,102,2,135,85,32,26,234,217,167,74]
        elif rnstream == 2:
            actionlist = [180,131,147,94,138,89,60,42,120,161,245,249,166,89,9,225,239,115,153,29,133,235,171,149,247,46,124,144,139,158,53,38,196,171,214]
        elif rnstream == 3:
            actionlist = [199,216,50,59,185,189,96,52,120,220,253,153,161,232,202,90,85,57,231,192,188,116,1,247,242,8,111,69,204,85,5,44,20,78,7]
        elif rnstream == 4:
            actionlist = [191,245,158,254,220,198,3,126,112,112,157,28,30,57,110,41,24,241,138,150,55,244,29,205,9,38,133,221,132,86,44,119,69,19,18]
        elif rnstream == 5:
            actionlist = [63,9,68,115,244,248,110,164,186,16,212,216,82,50,139,16,211,75,185,71,31,189,192,43,79,26,170,3,201,73,86,176,142,238,250]
        elif rnstream == 6:
            actionlist = [76,108,240,91,174,179,117,79,69,46,115,64,195,16,37,44,88,206,176,73,58,228,198,221,246,119,108,217,152,183,99,213,65,137,242]
        elif rnstream == 7:
            actionlist = [191,136,232,12,222,125,119,103,169,150,197,137,216,33,42,248,98,232,117,136,24,195,40,76,156,78,25,238,162,141,69,24,87,38,74]
        elif rnstream == 8:
            actionlist = [9,156,237,163,130,115,101,24,227,172,155,163,93,67,157,24,102,28,137,149,165,124,234,111,173,82,185,246,234,198,123,212,191,158,105]
        elif rnstream == 9:
            actionlist = [19,80,72,87,244,199,27,243,155,189,123,171,217,73,197,75,161,149,230,117,107,217,44,20,36,191,40,142,95,90,100,45,180,58,124]
    elif dataset == 4:
        if rnstream == 0:
            actionlist = [76,212,183,201,156,145,114,146,14,91,161,96,44,166,4,228,51,220,188,107,250,188,177,163,60,113,139,230,72,150,18,13,168,117,214]
        elif rnstream == 1:
            actionlist = [53,190,173,91,118,192,123,220,92,113,64,132,21,49,70,232,98,47,39,58,100,30,43,8,17,13,16,87,13,28,2,4,65,1,18]
        elif rnstream == 2:
            actionlist = [180,5,42,136,138,63,49,252,42,4,208,132,195,61,148,209,2,95,97,56,141,91,127,59,51,174,11,126,139,158,3,170,90,64,221]
        elif rnstream == 3:
            actionlist = [199,240,5,12,185,63,53,173,162,251,71,148,87,85,171,90,160,155,126,45,254,129,112,42,24,170,230,162,204,138,31,214,124,34,95]
        elif rnstream == 4:
            actionlist = [191,227,245,231,220,198,10,126,146,252,234,51,181,89,207,47,131,241,104,205,192,145,17,128,107,72,52,150,242,181,45,108,132,167,24]
        elif rnstream == 5:
            actionlist = [63,115,251,220,244,145,115,164,186,228,61,74,119,55,81,2,27,102,199,99,140,85,55,83,73,234,49,165,78,121,21,113,6,130,103]
        elif rnstream == 6:
            actionlist = [76,32,236,144,174,194,161,79,69,209,36,26,172,210,3,239,187,198,68,65,48,61,178,122,130,89,249,117,50,13,30,66,44,168,194]
        elif rnstream == 7:
            actionlist = [191,109,66,12,222,55,209,103,169,151,35,248,155,58,90,132,132,230,94,190,228,55,134,33,41,19,132,107,122,47,25,57,59,82,130]
        elif rnstream == 8:
            actionlist = [9,7,237,130,130,201,198,24,227,0,173,32,215,110,110,24,232,152,39,103,242,74,114,123,68,56,9,188,80,146,19,189,14,185,146]
        elif rnstream == 9:
            actionlist = [19,10,247,87,244,235,219,177,229,83,0,117,155,135,22,199,255,207,42,150,236,93,171,147,95,67,131,154,224,124,146,177,193,64,3]
    return actionlist[time]

class Simulation:
    """Based on the beer game by Chaharsooghi (2008)."""

    def __init__(self, seed, dataset):
        self.case = BeerGame()
        self.case.order_policy = "X+Y"
        self.case.divide = 1 
        # self.case.demand_dist = dataset
        # self.case.leadtime_dist = dataset
        self.seed = seed
        self.dataset = dataset
        random.seed(seed)
        possible_actions = [0, 1, 2, 3]
        self.no_actions = len(possible_actions) ** self.case.no_stockpoints

        # Initialize environment
        self.env = InventoryEnv(case=self.case,
                                action_low=0,
                                action_high=0,
                                action_min=0,
                                action_max=0,
                                state_low=0,
                                state_high=0,
                                coded=True,
                                fix=True,
                                ipfix=False,
                                method='Q-learning')


    def random_action(self):
        """Generate a random set of actions."""
        return random.randint(0, self.no_actions - 1)

    def get_next_action(self, time):
        """Determine the next action to be taken.

        Based on the exploitation rate
        Returns a list of actions
        """
        action = get_next_action(time, self.seed, self.dataset)
        return action

    def perform_simulation(self):
        """
        Perform the greedy policy.

        Hence, it does not use a random action, but always choses the best
        action according to the Q_table
        """

        totalreward, time = 0, 0
        holdinglist, bolist, policy = [], [], []
        _ = self.env.reset()
        while time < self.case.horizon:
            action = self.random_action()
            # action = self.get_next_action(time)
            policy.append(action)
            _, reward, _, info = self.env.simulate(decode_action(action))
            holdinglist.append(info['holding_costs'])
            bolist.append(-info['backorder_costs'])
            totalreward += reward
            time += 1
        print(totalreward)
        return totalreward, policy, holdinglist, bolist

STARTTIME = ct.time()
for sim in range(500):
    for dataset in range(1, 2):
        ENV = Simulation(sim, dataset)
        totalrewardlist, latest_policy, holdinglist, bolist = Simulation.perform_simulation(ENV)
        file = open("resultsNEW, dataset" + str(dataset) + ".txt", "a+")
        file.write("Total reward")
        file.write(str(totalrewardlist))
        file.write("\n")
        file.write("Policy seed")
        file.write(str(latest_policy))
        file.write("\n")
        file.write("holding list")
        file.write(str(holdinglist))
        file.write("\n")
        file.write("bo list")
        file.write(str(bolist))
        file.write("\n")
        file.close()
