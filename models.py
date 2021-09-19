import math
import torch.nn as nn
import torch.nn.functional as F
import copy

from config import *

h = 32

class DQN(nn.Module):
    def __init__(self, recent_k, n_agents, n_actions, initial_Q):
        super(DQN, self).__init__()
        # self.linear1 = nn.Linear(n_actions*n_agents, h)
        self.linear1 = nn.Linear(recent_k*n_agents, h)
        # torch.nn.init.xavier_uniform_(self.linear1.weight)
        # torch.nn.init.zeros_(self.linear1.weight)
        # torch.nn.init.zeros_(self.linear1.bias)

        # self.linear2 = nn.Linear(h, h)
        # torch.nn.init.xavier_uniform_(self.linear2.weight)
        # torch.nn.init.zeros_(self.linear2.bias)
        #
        # self.linear3 = nn.Linear(h, h)
        # torch.nn.init.xavier_uniform_(self.linear3.weight)
        # torch.nn.init.zeros_(self.linear3.bias)

        self.linear2 = nn.Linear(h, n_actions)
        # torch.nn.init.xavier_uniform_(self.linear4.weight)
        # torch.nn.init.zeros_(self.linear4.bias)
        # torch.nn.init.zeros_(self.linear4.weight)
        # self.linear4.bias = torch.nn.Parameter(copy.deepcopy(initial_Q))

    def forward(self, x):
        x = x.view(-1, recent_k * n_agents)
        x = actions_space[x.long()] / actions_space[-1]
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
