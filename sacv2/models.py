import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Tuple


class ValueNetwork(nn.Module):

    # Initialize network arhitectures
    def __init__(self, input_dim: int, output_dim:int, init_w: float=3e-3):
        super(ValueNetwork, self).__init__()
        # The format is (input size, output size)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

        # Initialize network parameters with uniform random
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor)->torch.Tensor:
        # Giving value to the network, return the output 
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class SoftQNetwork(nn.Module):
    
    # Initialize network arhitectures
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int=256, init_w: int=3e-3):
        super(SoftQNetwork, self).__init__()
        # The format is (input size, output size)
        self.linear1 = nn.Linear(input_dim + output_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        # Initialize network parameters with uniform random
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    # Giving state+action to the network, return the Q value
    def forward(self, state, action):
        # Concatenate state and action
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    
    # Initialize network arhitectures
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int=256, init_w: float=3e-3, log_std_min: int=-20, log_std_max: int=2):
        super(PolicyNetwork, self).__init__()
        # The format is (input size, output size)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        # Initialize network parameters with uniform random
        self.mean_linear = nn.Linear(hidden_size, output_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, output_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: float)-> Tuple[torch.Tensor, torch.Tensor]:
        # Giving value to the network, return the output as the mean and variance for the action
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state: torch.Tensor, scale: float, epsilon: float=1e-6)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sampling the actions, log_pi, mean, std for the given states input -> reparemeterization trick details are given in the SAC paper appendix
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Make a normal distribution from the given mean and variance
        normal = Normal(mean, std)
        
        # take a sample from the normal distribution
        z = normal.rsample()

        # Using the reparameterization trick, obtain the action from the sampled z
        action = torch.tanh(z)

        log_pi = normal.log_prob(z) -  torch.log(scale[0] *(1 - action.pow(2)) + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi, mean, std