import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_nodes=64, fc2_nodes=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.fc1 = nn.Linear(state_size, fc1_nodes)
        self.fc2 = nn.Linear(fc1_nodes, fc2_nodes)
        self.fc3 = nn.Linear(fc2_nodes, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        fc1_relu=F.relu(self.fc1(state))
        fc2_relu=F.relu(self.fc2(fc1_relu))
        return self.fc3(fc2_relu)
