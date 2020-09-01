import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Critic(nn.Module):

    # Initialize network arhitectures
    def __init__(self, obs_dim: int, action_dim: int):
        super(Critic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # The format is (input size, output size)
        self.linear1 = nn.Linear(self.obs_dim, 400)
        self.linear2 = nn.Linear(400 + self.action_dim, 300)
        self.linear4 = nn.Linear(300, 1)
    
    # Giving state+action to the network, return the Q value
    def forward(self, x: torch.Tensor, a: torch.Tensor)-> torch.Tensor:
        x = F.relu(self.linear1(x))

        # Concatenate x and a
        xa_cat = torch.cat([x,a], 1)

        xa = F.relu(self.linear2(xa_cat))
        qval = self.linear4(xa)

        return qval

class Actor(nn.Module):

    # Initialize network arhitectures
    def __init__(self, obs_dim: int, action_dim: int):
        super(Actor, self).__init__()

        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # The format is (input size, output size)
        self.linear1 = nn.Linear(self.obs_dim, 400)
        self.linear2 = nn.Linear(400, 300)
        self.linear3 = nn.Linear(300, self.action_dim)

    # Giving state to the network, return the action
    def forward(self, obs: torch.Tensor)-> torch.Tensor:
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x
