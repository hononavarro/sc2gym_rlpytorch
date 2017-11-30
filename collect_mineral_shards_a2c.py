import copy
import glob
import os
import time

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

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
from kfac import KFACOptimizer
from model import CNNPolicy, MLPPolicy
from storage import RolloutStorage
from visualize import visdom_plot

FLAGS = flags.FLAGS
FLAGS([__file__])


_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_NO_OP = 0

_ENV_NAME = "SC2CollectMineralShards-v0"
_VISUALIZE = True
_STEP_MUL = None
_NUM_EPISODES = 10000


class CollectMineralShards_A2C:


    def __init__(self, env_name, visualize=False, step_mul=None) -> None:

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

    NUM_AGENTS = 1

    envs = [CollectMineralShards_A2C(_ENV_NAME,_VISUALIZE, _STEP_MUL) for i in range(NUM_AGENTS)]





    # example = CollectMineralShards_A2C(_ENV_NAME,_VISUALIZE, _STEP_MUL)
    # rewards = example.run(_NUM_EPISODES)
    # print('Total reward: {}'.format(rewards.sum()))
    # print('Average reward: {} +/- {}'.format(rewards.mean(), rewards.std()))
    # print('Minimum reward: {}'.format(rewards.min()))
    # print('Maximum reward: {}'.format(rewards.max()))

if __name__ == "__main__":
    main()
