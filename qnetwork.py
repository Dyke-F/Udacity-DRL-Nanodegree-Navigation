import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNetwork(nn.Module):
    """ Create a Deep Q-Learning Policy Network """
    
    def __init__(self, state_size, action_size, seed, fc1_units=40, fc2_units=10):
        """ Initialize Network with ...
        ==========
        PARAMETERS
        ==========
            state_size (integer) = Observation space size
            action_size (integer) = Number of available actions per state
            seed (integer) = Random seed to maintain reproducibility
            fc1_units, fc2_units (integer) = Number of nodes in hidden layers
        """
        
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        
    def forward(self, x):
        """ Perform a single forward propagation through the network,
        mappuingstate to action values.
        ==========
        Parameters
        ==========
            x = Observation space
        """
        
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x