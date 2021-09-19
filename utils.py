import random
from collections import namedtuple
import torch.nn.functional as F
from config import *

import numpy as np

random.seed(12345)
np.random.seed(12345)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class PERMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = torch.zeros(capacity, device=device)

    def push(self, defect, *args):
        """Saves a transition."""
        # max_priori = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.priorities[self.position] = defect
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, steps_done):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]


        rank = torch.unique(prios, sorted=True, return_inverse=True)[1].float()
        probs = torch.exp(rank*LAM)
        probs = probs/probs.sum()

        if steps_done%50000 == 0:
            print('Sampling steps', steps_done.data, 'Cur Poi', self.position, 'Probs Max:', probs.argmax(), probs.max(),
                  'Min:', probs.argmin(), probs.min())
            print('Max Memory Defect', self.priorities.max(), 'at', self.priorities.argmax(),
                  'Min', self.priorities.min(), 'at', self.priorities.argmin())
        # indices = rank[-batch_size:]
        indices = probs.multinomial(num_samples=batch_size, replacement=True)
        samples = [self.memory[idx] for idx in indices]
        # samples = random.sample(self.memory, batch_size)
        weights = torch.ones(batch_size, device=device)/batch_size
        return samples, weights

    # def update_priorities(self, batch_indices, batch_priorities):
    #     for idx, prio in zip(batch_indices, batch_priorities):
    #         self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)


def reward_comp(action):
    # Compute profits for all agents
    # Input: actions taken by all agents, shape: n_agents;
    # Output: profits for all agents
    action = action.long()
    price = actions_space[action]
    demand = torch.exp((quality - price) / horizon)
    demand = demand / (torch.sum(demand) + torch.exp(a0 / horizon))
    reward = torch.mul(price - margin_cost, demand)
    return reward.view(1, -1)

def select_action_deep(policy_net, state, memory, steps_done):
    sample = random.random()
    eps_threshold = torch.exp(-eps * steps_done)

    if len(memory) >= BATCH_SIZE and sample > eps_threshold:
        with torch.no_grad():
            batch_state = state.unsqueeze(0)
            return policy_net(batch_state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.float)


def optimize_model(agent, policy_net, target_net, memory, optimizer, steps_done):
    if len(memory) < BATCH_SIZE:
        return

    transitions, weights = memory.sample(BATCH_SIZE, steps_done)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    next_states = torch.stack(batch.next_state)
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action).long()
    reward_batch = torch.cat(batch.reward)


    state_action_values = policy_net(state_batch).gather(1, action_batch[:, agent].view(-1, 1))
    next_state_values = target_net(next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch[:, agent]

    loss = weights*(state_action_values.squeeze(1) - expected_state_action_values)**2
    loss = loss.mean()
    # loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss
