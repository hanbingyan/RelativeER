from collections import namedtuple
import numpy as np
from scipy.stats import rankdata
import pickle

class bcolors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PINK = '\033[35m'
    GREY = '\033[36m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
MEM_SIZE = 1000
BATCH_SIZE = 8
GAMMA = 0.95
LAM = 0.2

num_sub = 800000
eps = 1e-6
alpha = 0.15

n_instance = 5
n_agents = 6
actions_space = np.arange(1.0, 2.5, 0.3)
quality = np.ones(n_agents) * 2
margin_cost = np.ones(n_agents)
horizon = 1 / 4
a0 = 0
n_actions = actions_space.size

state_ravel = (n_actions,) * n_agents

ss = np.random.SeedSequence(12345)
# Last a few rngs for sampling memories
child_seeds = ss.spawn(n_agents * 2)
rng = [np.random.default_rng(s) for s in child_seeds]


class PERMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros(capacity)

    def push(self, defect, *args):
        """Saves a transition."""
        max_priori = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.priorities[self.position] = defect
        self.position = (self.position + 1) % self.capacity

    def sample(self, agent, batch_size):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]

            # Nash case
        #         probs = np.minimum(1e10, np.maximum(1e-9, np.exp(0.1 - rankdata(prios, method='min')/MEM_SIZE/temper)))
        # Collusion case
        probs = np.minimum(1e10, np.maximum(1e-9, np.exp(rankdata(prios, method='min') * LAM)))
        probs = probs / probs.sum()
        indices = rng[agent + n_agents].choice(len(probs), size=batch_size, p=probs)
        #         indices = np.random.binomial(1, p=0, size=1)
        #         indices = [0]
        samples = [self.memory[idx] for idx in indices]

        if steps_done % 10000 == 0:
            print(bcolors.RED + 'Step', steps_done, 'Memory priority min', self.priorities.min(), 'is at position',
                  self.priorities.argmin(), bcolors.ENDC)
            print(bcolors.RED + 'Memory priority max', self.priorities.max(), 'is at position',
                  self.priorities.argmax(), bcolors.ENDC)
            print('Sampled priority is', self.priorities[indices[0]])
        # print('Max prob at', probs.argmax(), probs.max(), 'Min:', probs.argmin(), probs.min())

        weights = np.ones_like(indices) / batch_size
        return samples, indices, weights

    def __len__(self):
        return len(self.memory)


def replay_classic_reward(action):
    # Compute profits for all agents
    price = actions_space[action]
    demand = np.exp((quality - price) / horizon)
    demand = demand / (np.sum(demand) + np.exp(a0 / horizon))
    reward = np.multiply(price - margin_cost, demand)
    return reward


def replay_classic_select(agent, state, eps_threshold):
    #     sample = random.random()
    if rng[agent].random() > eps_threshold:
        return Q[agent][state].argmax()
    else:
        return rng[agent].integers(0, n_actions, 1, dtype=int)


def replay_classic_opt(agent, mem, BS):
    if len(mem) < MEM_SIZE:
        return

    transitions, indices, weights = mem.sample(agent, BS)
    batch = Transition(*zip(*transitions))

    if steps_done % 10000 == 0:
        print('Sampled States', np.unravel_index(batch.state[0], state_ravel))

    delta = np.zeros(BS)
    for i in range(BS):
        stat = batch.state[i]
        next_stat = batch.next_state[i]
        act = batch.action[i]
        rew = batch.reward[i]
        delta[i] = rew[agent] + GAMMA * Q[agent, next_stat].max() - Q[agent, stat, act[agent]]
        Q[agent, stat, act[agent]] = Q[agent, stat, act[agent]] + alpha * weights[i] * delta[i]
    return


def simulate(last_state, N):
    state = last_state
    action = np.zeros(n_agents, dtype=int)
    reward = np.zeros(n_agents)
    for k in range(N):
        # For each agent, select and perform an action
        for i in range(n_agents):
            action[i] = replay_classic_select(i, state, 0)

        reward += replay_classic_reward(action)
        # Move to the next state
        state = np.ravel_multi_index(action, state_ravel)
        if k > N - 10:
            print(np.unravel_index(state, state_ravel))
    return reward / N


Q_hist = []
end_price = []

Qmax_hist = []
Qmin_hist = []

for sess in range(n_instance):
    steps_done = 0

    state_hist = np.zeros(1000, dtype=int)
    # Initialize the environment and state
    init = np.zeros(n_agents, dtype=int)
    for i in range(n_agents):
        init[i] = rng[i].integers(0, n_actions, size=1)
    state = np.ravel_multi_index(init, state_ravel)
    #     state = rng.integers(0, n_actions**n_agents, size=1)
    state_hist[0] = state
    # Counter for variations in heat
    count = 0

    Defect = np.zeros((n_agents, n_actions ** n_agents, n_actions))
    Q = np.zeros((n_agents, n_actions ** n_agents, n_actions))

    memo_list = []
    for i in range(n_agents):
        memo_list.append(PERMemory(MEM_SIZE))

    for i_episode in range(num_sub):
        # For each agent, select and perform an action
        action = np.zeros(n_agents, dtype=int)
        eps_threshold = np.exp(-eps * steps_done)

        for i in range(n_agents):
            action[i] = replay_classic_select(i, state, eps_threshold)

        steps_done += 1

        reward = replay_classic_reward(action)

        r_rank = rankdata(reward, method='min')
        for agent in range(n_agents):
            Defect[agent, state, action[agent]] += r_rank.max() - r_rank[agent]

        # if action[0] - margin_cost[0] > action[1] - margin_cost[1]:
        #     Defect[0, state, action[0]] += 1
        # elif action[0] - margin_cost[0] < action[1] - margin_cost[1]:
        #     Defect[1, state, action[1]] += 1

        # Observe new state
        next_state = np.ravel_multi_index(action, state_ravel)

        # Store the transition in memory
        for i in range(n_agents):
            memo_list[i].push(Defect[i, state, action[i]], state, action, next_state, reward)

        old_heat = Q.argmax(2)
        old_Qmax = Q.max()
        for i in range(n_agents):
            replay_classic_opt(i, memo_list[i], BATCH_SIZE)

        new_heat = Q.argmax(2)
        new_Qmax = Q.max()

        if np.abs(old_Qmax - new_Qmax) < 1e-5 and np.sum(np.abs(old_heat - new_heat)) == 0:
            count += 1
        else:
            count = 0

        state = next_state
        state_hist[steps_done % 1000] = state

        if i_episode % 1000 == 0:
            print(bcolors.GREEN + 'Instance', sess, 'Count', count, 'Steps done:', steps_done, bcolors.ENDC)
            uniq0, freq0 = np.unique(np.unravel_index(state_hist, state_ravel)[0], return_counts=True)
            fq0 = freq0 / freq0.sum()
            idx0 = np.argsort(fq0)[-4:]
            print(bcolors.PINK + 'Highest freq0', fq0[idx0], bcolors.ENDC)
            print(bcolors.PINK + 'Price', actions_space[uniq0[idx0].astype(int)], bcolors.ENDC)

            uniq1, freq1 = np.unique(np.unravel_index(state_hist, state_ravel)[1], return_counts=True)
            fq1 = freq1 / freq1.sum()
            idx1 = np.argsort(fq1)[-4:]
            print(bcolors.PINK + 'Highest freq1', fq1[idx1], bcolors.ENDC)
            print(bcolors.PINK + 'Price', actions_space[uniq1[idx1].astype(int)], bcolors.ENDC)

            #             uniq2, freq2 = np.unique(np.unravel_index(state_hist, state_ravel)[2], return_counts=True)
            #             fq2 = freq2 / freq2.sum()
            #             idx2 = np.argsort(fq2)[-4:]
            #             print(bcolors.PINK + 'Highest freq2', fq2[idx2], bcolors.ENDC)
            #             print(bcolors.PINK + 'Price', actions_space[uniq2[idx2].astype(int)], bcolors.ENDC)

            print('Q max', Q.max(), 'Q min', Q.min())
            print('Defect matrix max', Defect.max(), 'is at',
                  np.unravel_index(np.argmax(Defect, axis=None), Defect.shape))
            # print('Defect min', Defect.min(), np.unravel_index(np.argmin(Defect, axis=None), Defect.shape))
            # print('Simulated reward', simulate(1000))
            Qmax_hist.append(Q.max())
            Qmin_hist.append(Q.min())

        if count == 100000:
            print('Terminate condition satisfied with price', np.array(state_hist[-20:]))
            break
    end_price.append(state_hist[-20:])
    Q_hist.append(Q)

    print(np.array(state_hist[-20:]))

with open('ThreeAgents_LAM002_Q.pickle', 'wb') as fp:
    pickle.dump(Q_hist, fp)

with open('ThreeAgents_LAM002_price.pickle', 'wb') as fp:
    pickle.dump(end_price, fp)


# N = 200000
# monopoly = replay_classic_reward([18, 18])[0]
# nash = replay_classic_reward([7, 7])[0]
#
# ratio = np.zeros((n_instance, n_instance))
#
# for agent0 in range(n_instance):
#     for agent1 in range(n_instance):
#         sim_Q = np.zeros((n_agents, n_actions**n_agents, n_actions))
#         sim_Q[0, :, :] = Q_hist[agent0][0, :, :]
#         sim_Q[1, :, :] = Q_hist[agent1][1, :, :]
# #         state = [7, 7]
#         init = np.zeros(n_agents, dtype=int)
#         for i in range(n_agents):
#             init[i] = rng[i].integers(0, n_actions, size=1)
#         state = np.ravel_multi_index(init, state_ravel)
#
#         action = np.zeros(n_agents, dtype=int)
#         reward = np.zeros(n_agents)
#         for k in range(N):
#             # For each agent, select and perform an action
#             for i in range(n_agents):
#                 action[i] = sim_Q[i, state].argmax()
#             if k > N - 10000:
#                 reward += replay_classic_reward(action)
#             # Move to the next state
#             state = np.ravel_multi_index(action, state_ravel)
#             avg = np.sum(reward)/10000/n_agents
#             ratio[agent0, agent1] = (avg - nash)/(monopoly - nash)
#         print('Instance', agent0, 'vs', agent1, avg, 'ratio', ratio[agent0, agent1])
#
# with open('ThreeAgents_ratio.pickle', 'wb') as fp:
#     pickle.dump(ratio, fp)
