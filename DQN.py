import torch.optim as optim
import copy
import pickle

from utils import *
from models import *

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

heat_record = []
act_record = []

for ins in range(num_instance):
    initial_Q = torch.zeros(n_actions, device=device)

    Defect = torch.zeros(n_agents, n_actions, n_actions, n_actions, device=device)
    policy_net = []
    target_net = []
    optimizer = []
    memory = []
    for i in range(n_agents):
        memory.append(PERMemory(MEM_SIZE))
        policy_net.append(DQN(recent_k, n_agents, n_actions, initial_Q).to(device))
        target_net.append(DQN(recent_k, n_agents, n_actions, initial_Q).to(device))
        target_net[i].load_state_dict(policy_net[i].state_dict())
        target_net[i].eval()
        optimizer.append(optim.Adam(policy_net[i].parameters(), lr=1e-4))

    act_hist = torch.zeros(1000, n_agents, device=device)
    Q_hist = torch.zeros(2, n_agents, n_actions ** 2, device=device)

    Qmax_hist = []
    Qmin_hist = []
    loss_hist = []

    test_action = torch.zeros(n_agents, n_actions ** 2, device=device)
    steps_done = torch.zeros(1, device=device)
    # Initialize the environment and state
    heat_episode = torch.zeros(num_sub, n_agents, n_actions, n_actions, device=device)
    heat = torch.zeros(n_agents, n_actions, n_actions, device=device)

    state = torch.randint(0, n_actions, size=(recent_k, n_agents), dtype=torch.float,
                          device=device)

    count = 0
    old0_max = 0
    old1_max = 0
    for t in range(num_sub):
        # For each agent, select and perform an action
        action = torch.zeros(1, n_agents, device=device)
        for i in range(n_agents):
            action[0, i] = select_action_deep(policy_net[i], state, memory[i], steps_done)

        steps_done += 1
        reward = reward_comp(action.view(-1))


        act_hist[int(t % 1000), :] = action
        # Observe new state
        next_state = torch.cat((state[1:], action.view(1, -1)), dim=0)

        if reward[0, 0] < reward[0, 1]:
            Defect[0, state[0, 0].long(), state[0, 1].long(), action[0, 0].long()] += 1
        elif reward[0, 0] > reward[0, 1]:
            Defect[1, state[0, 0].long(), state[0, 1].long(), action[0, 1].long()] += 1

        if t%50000 == 0:
            print('Defect matrix max', Defect.max(), 'Min', Defect.min())

        for i in range(n_agents):
            memory[i].push(Defect[i, state[0, 0].long(), state[0, 1].long(), action[0, i].long()], state, action,
                           next_state, reward)

        # Perform one step of the optimization (on the target network)
        # befopt_dict0 = copy.deepcopy(policy_net[0].state_dict())
        # befopt_dict1 = copy.deepcopy(policy_net[1].state_dict())
        for i in range(n_agents):
            loss = optimize_model(i, policy_net[i], target_net[i], memory[i], optimizer[i], steps_done)


        # Move to the next state
        state = copy.deepcopy(next_state)
        # print('Parameter diff', torch.sum(torch.abs(aftopt_dict0['linear3.bias'] - befopt_dict0['linear3.bias'])))

        for k in range(n_agents):
            Q_hist[int(t % 2), k, :] = policy_net[k](test_batch).max(1)[0].detach()
            test_action[k, :] = policy_net[k](test_batch).max(1)[1].detach()
            heat[k, :, :] = test_action[k, :].view(n_actions, n_actions)



        Qmax_hist.append(Q_hist[int(t % 2)].max())
        Qmin_hist.append(Q_hist[int(t % 2)].min())
        heat_episode[t, :, :, :] = heat

        if t > BATCH_SIZE:
            loss_hist.append(loss.data)

        # Update the target network, copying all weights and biases in DQN
        if t % TARGET_UPDATE == 0:
            for i in range(n_agents):
                target_net[i].load_state_dict(policy_net[i].state_dict())

        if (t + 1000) % 10000 == 0:
            print(bcolors.RED + 'Instance', ins, 'steps done:', steps_done.item(), bcolors.ENDC)
            print('Action before optimization', action)
            print('loss', loss.data)

            uniq0, freq0 = torch.unique(act_hist[:, 0], sorted=True, return_counts=True)
            # heat_unique0.append(uniq0.to('cpu').numpy())
            # heat_freq0.append(freq0.to('cpu').numpy())

            uniq1, freq1 = torch.unique(act_hist[:, 1], sorted=True, return_counts=True)
            # heat_unique1.append(uniq1.to('cpu').numpy())
            # heat_freq1.append(freq1.to('cpu').numpy())

            # uniq0, freq0 = torch.unique(heat_episode[(t-1000):t, 0, :, :], sorted=True, return_counts=True)
            # heat_unique0.append(uniq0.to('cpu').numpy())
            # heat_freq0.append(freq0.to('cpu').numpy())

            # uniq1, freq1 = torch.unique(heat_episode[(t-1000):t, 1, :, :], sorted=True, return_counts=True)
            # heat_unique1.append(uniq1.to('cpu').numpy())
            # heat_freq1.append(freq1.to('cpu').numpy())
            uniq0_max = uniq0[freq0.argmax()]
            uniq1_max = uniq1[freq1.argmax()]

            f0_max = freq0.max()/freq0.sum()
            f1_max = freq1.max()/freq1.sum()

            print('Agent 0 act unique', uniq0)
            print('Agent 0 most act', uniq0_max, 'with freq', f0_max)
            print('Agent 1 act unique', uniq1)
            print('Agent 1 most act', uniq1_max, 'with freq', f1_max)

            if (f0_max + f1_max)/n_agents > 0.9 and uniq0_max - old0_max ==0 and uniq1_max - old1_max == 0:
                count += 1
            else:
                count = 0

            old0_max = uniq0_max
            old1_max = uniq1_max

            print('Q max', Q_hist.max(), 'Q min', Q_hist.min(), 'count', count)

            if count == 20:
                print('Terminate condition satisfied')
                break
    heat_record.append(heat.to('cpu').numpy())
    act_record.append(act_hist.to('cpu').numpy())

with open('DQN_LAM5_heat.pickle', 'wb') as fp:
    pickle.dump(heat_record, fp)

with open('DQN_LAM5_act.pickle', 'wb') as fp:
    pickle.dump(act_record, fp)



# plt.figure(figsize=(8, 6))
# ax = sns.heatmap(heat[0].cpu().numpy(), annot=True)
# plt.show()
#
# plt.figure(figsize=(8, 6))
# ax = sns.heatmap(heat[1].cpu().numpy(), annot=True)
# plt.show()
#
# plt.figure(figsize=(8, 6))
# plt.plot(torch.tensor(loss_hist).cpu().numpy()[::2])
# plt.show()
#
# plt.figure(figsize=(8, 6))
# plt.plot(torch.tensor(Qmax_hist).cpu().numpy())
# plt.plot(torch.tensor(Qmin_hist).cpu().numpy())
# plt.show()
