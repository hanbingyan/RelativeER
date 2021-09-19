import torch

torch.manual_seed(12345)
# check gpu is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
GAMMA = 0.95
MEM_SIZE = 2000
BATCH_SIZE = 16

TARGET_UPDATE = 100
LAM = 5
eps = 1e-5

actions_space = torch.arange(1.2, 2.0, 0.04, device=device)
n_actions = actions_space.size(0)
n_agents = 2
recent_k = 1

quality = torch.ones(n_agents, device=device) * 2
margin_cost = torch.ones(n_agents, device=device)
horizon = torch.ones(1, device=device) / 4
a0 = 0

num_instance = 5
num_sub = 1000000

# Batch used for heatmap computing
test_batch = torch.zeros(n_actions**n_agents, recent_k, n_agents, device=device)
for i in range(n_actions**n_agents):
    test_batch[i, 0, 0] = i // (n_actions ** (n_agents - 1))
    # test_batch[i, 0, 1] = i // (n_actions ** (n_agents - 2))
    test_batch[i, 0, 1] = i % n_actions
