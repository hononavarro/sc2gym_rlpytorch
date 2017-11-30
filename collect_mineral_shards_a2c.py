import sys

import inspect
from storage import RolloutStorage

import math
import random
from collections import namedtuple
import numpy as np
import gym
from absl import flags
from models import a2c
import torch.optim as optim


# noinspection PyUnresolvedReferences
import sc2gym.envs

FLAGS = flags.FLAGS
FLAGS([__file__])


_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_NO_OP = 0

_ENV_NAME = "SC2CollectMineralShards-v0"
_VISUALIZE = True
_STEP_MUL = None
_NUM_EPISODES = 10000
_CUDA=True


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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

class CollectMineralShards_A2C:


    def __init__(self, env_name, visualize=False, step_mul=None) -> None:

        print("creating env....")
        self.env_name = env_name
        self.visualize = visualize
        self.step_mul = step_mul

    def run(self, num_episodes=1):
        env = gym.make(self.env_name)

        env.settings['visualize'] = self.visualize
        env.settings['step_mul'] = self.step_mul

        #episode_rewards = np.zeros((num_episodes, ), dtype=np.int32)
        for ix in range(num_episodes):
            obs = env.reset()
            t = 0
            done = False
            while not done:
                action = self.get_action(env, obs)


                new_obs, reward, done, _ = env.step(action)

                #reward = Tensor([reward])

                memory.push(obs, action, new_obs, reward)
                obs = new_obs

                optimize_model()
                t +=1

            episode_rewards.append(reward)
            #plot_rewards()

        env.close()

        return episode_rewards

    def get_action(self, env, obs):
        neutral_y, neutral_x = (obs[0] == _PLAYER_NEUTRAL).nonzero()
        player_y, player_x = (obs[0] == _PLAYER_FRIENDLY).nonzero()
        if not neutral_y.any():
            raise Exception('No minerals found!')
        if not player_y.any():
            raise Exception('No marines found!')

        #player = [np.ceil(player_x.mean()).astype(int), np.ceil(player_y.mean()).astype(int)]
        #shards = np.array(list(zip(neutral_x, neutral_y)))
        #closest_ix = np.argmin(np.linalg.norm(np.array(player) - shards, axis=1))
        #target = shards[closest_ix]

        target = select_action(obs[0])


        return target


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                              math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(
            Variable(torch.from_numpy(np.array(state).reshape((64,64))).unsqueeze(0).unsqueeze(0), volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1).cpu().numpy().squeeze()
    else:
        return np.random.randint(1024)

def main():

    NUM_AGENTS = 4
    NUM_FRAMES_TO_STACK = 1
    LR=7e-4
    EPS=1e-5
    ALPHA=0.99
    NUM_STEPS=10000


    #TODO this should be given by env
    obs_shape = [64, 64]
    action_space = 1024

    RECURRENT_POLICY=False
    ACTION_SPACE = obs_shape[0]*obs_shape[1]

    envs = [CollectMineralShards_A2C(_ENV_NAME, _VISUALIZE, _STEP_MUL) for i in range(NUM_AGENTS)]

    obs_shape = (obs_shape[0] * NUM_FRAMES_TO_STACK, *obs_shape[1:])

    actor_critic = a2c.CNNPolicy(obs_shape[0], ACTION_SPACE, RECURRENT_POLICY)
    optimizer = optim.RMSprop(actor_critic.parameters(), LR, eps=EPS, alpha=ALPHA)


    rollouts = RolloutStorage(NUM_STEPS, NUM_AGENTS, obs_shape, action_space, actor_critic.state_size)

    sys.exit()



    # if len(envs.observation_space.shape) == 3:
    #     actor_critic = CNNPolicy(obs_shape[0], envs.action_space, args.recurrent_policy)


    if _CUDA:
        actor_critic.cuda()

    optimizer = optim.RMSprop(model.parameters())
    memory = ReplayMemory(10000)
    steps_done = 0



    # example = CollectMineralShards_A2C(_ENV_NAME,_VISUALIZE, _STEP_MUL)
    # rewards = example.run(_NUM_EPISODES)
    # print('Total reward: {}'.format(rewards.sum()))
    # print('Average reward: {} +/- {}'.format(rewards.mean(), rewards.std()))
    # print('Minimum reward: {}'.format(rewards.min()))
    # print('Maximum reward: {}'.format(rewards.max()))

if __name__ == "__main__":
    main()
