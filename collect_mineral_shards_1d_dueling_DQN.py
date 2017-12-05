
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import gym
from absl import flags

# noinspection PyUnresolvedReferences
import sc2gym.envs
import copy


FLAGS = flags.FLAGS
FLAGS([__file__])


_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_NO_OP = 0

_ENV_NAME = "SC2CollectMineralShards-v0"
_VISUALIZE = False
_STEP_MUL = None
_NUM_EPISODES = 30000


# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

#update period
UPDATE_PERIOD = 10000

PLOT_GRAPHS = False

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.hidden = nn.Linear(3200, 1024)
        self.head = nn.Linear(1024, 256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.hidden(x.view(x.size(0), -1)))
        return self.head(x)

class Dueling_DQN(nn.Module):

    def __init__(self):
        super(Dueling_DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)

        #state function
        self.state_hidden = nn.Linear(3200, 1024)
        self.state_head = nn.Linear(1024, 1)

        #advantage function
        self.adv_hidden = nn.Linear(3200, 1024)
        self.action_head = nn.Linear(1024, 256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        state_hidden = F.relu(self.state_hidden(x.view(x.size(0), -1)))
        state_value = self.state_head(state_hidden)

        action_hidden = F.relu(self.adv_hidden(x.view(x.size(0), -1)))
        action_value = self.action_head(action_hidden)

        return state_value + (action_value - action_value.mean())



#model = DQN()
model = Dueling_DQN()
#target_model = None

if use_cuda:
    model.cuda()

optimizer = optim.RMSprop(model.parameters())
memory = ReplayMemory(10000)
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                              math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(
            Variable(torch.from_numpy(np.array(state).reshape((16,16))).unsqueeze(0).unsqueeze(0), volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1).cpu().numpy().squeeze()
    else:
        return np.random.randint(256)

n_avg_samples = 100
episode_rewards = []
means = [0]*(n_avg_samples-1)

def plot_rewards():

    plt.figure(2)
    plt.clf()
    #rewards_t = torch.FloatTensor(episode_rewards)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(episode_rewards)
    # Take 100 episode averages and plot them too
    if len(episode_rewards) >= n_avg_samples:
        means.append(np.mean(np.array(episode_rewards[-n_avg_samples-1:-1])))
        plt.plot(means)

    plt.pause(0.001)  # pause a bit so that plots are updated



class CollectMineralShards1d_DQN:

    def __init__(self, env_name, visualize=False, step_mul=None) -> None:
        self.env_name = env_name
        self.visualize = visualize
        self.step_mul = step_mul
        self.env = gym.make(self.env_name)
        self.env.settings['visualize'] = self.visualize
        self.env.settings['step_mul'] = self.step_mul

    def run(self, num_episodes=1):
        global ALGORITHM
        global episode_rewards, reward_per_episode,means
        #global target_model
        total_steps = 0

        for ALGORITHM in [0]:
            reward_per_episode = []
            #episode_rewards = np.zeros((num_episodes, ), dtype=np.int32)
            for ix in range(num_episodes):
                obs = self.env.reset()
                t = 0
                done = False
                while not done:
                    #maybe update target network
                    # if total_steps % UPDATE_PERIOD == 0:
                    #     target_model = copy.deepcopy(model)
                    #     if use_cuda:
                    #         target_model.cuda()

                    action = self.get_action(self.env, obs)
                    new_obs, reward, done, _ = self.env.step(action)

                    memory.push(obs, action, new_obs, reward)
                    obs = new_obs
                    if ALGORITHM == 0:
                        optimize_model()
                    t +=1
                    total_steps += 1
                    reward_per_episode.append(reward)

                episode_rewards.append(np.sum(np.array(reward_per_episode)))
                reward_per_episode = []
                if PLOT_GRAPHS:
                    plot_rewards()

            np.save("episodeReward_DDQN"+str(ALGORITHM),np.array(episode_rewards))
            np.save("episodeRewardMean100_DDQN" + str(ALGORITHM),np.array(means))
            episode_rewards = []
            means = []

        self.env.close()

        return episode_rewards

    def get_action(self, env, obs):

        #if not neutral_y.any():
        #    raise Exception('No minerals found!')

        target = [0,0]
        if(ALGORITHM == 0):
            target = select_action(obs[0])
        elif(ALGORITHM == 1):
            target = np.random.randint(256)
        elif (ALGORITHM == 2):
            neutral_y, neutral_x = (obs[0] == _PLAYER_NEUTRAL).nonzero()
            player_y, player_x = (obs[0] == _PLAYER_FRIENDLY).nonzero()
            player = [np.ceil(player_x.mean()).astype(int), np.ceil(player_y.mean()).astype(int)]
            shards = np.array(list(zip(neutral_x, neutral_y)))
            closest_ix = np.argmin(np.linalg.norm(np.array(player) - shards, axis=1))
            target = np.ravel_multi_index(shards[closest_ix], obs.shape[1:])
        return target


last_sync = 0
c = 1000
updates = 0

def optimize_model():
    global last_sync
    global updates
    #global target_model

    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat(torch.from_numpy(np.array([s for s in batch.next_state
                                                if s is not None]))).unsqueeze(1),
                                     volatile=True).type(FloatTensor)
    state_batch = Variable(torch.cat(torch.from_numpy(np.array(batch.state))).unsqueeze(1)).type(FloatTensor)
    action_batch = Variable(torch.from_numpy(np.array(batch.action).transpose()).unsqueeze(1)).type(LongTensor)
    reward_batch = Variable(torch.from_numpy(np.array(batch.reward))).type(FloatTensor)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    check = model(state_batch)

    state_action_values = check.gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))



    #if updates % c == 0:
    #    print("Entro Updates")
    #    model_target = copy.deepcopy(model)
    #    print("Salgo Updates")

    updates += 1


    # next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]

    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def main():

    example = CollectMineralShards1d_DQN(_ENV_NAME,_VISUALIZE, _STEP_MUL)

    example.run(_NUM_EPISODES)


if __name__ == "__main__":
    main()
