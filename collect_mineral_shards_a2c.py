import sys

import inspect
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
import time



from models.storage_aux import RolloutStorage
from models.a2c import CNNPolicy
from torch.autograd import Variable


import math
import random
from collections import namedtuple
import numpy as np
import gym
import torch
from absl import flags
from models import a2c
import torch.optim as optim


# noinspection PyUnresolvedReferences
from sc2gym import envs

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

    def __call__(self):
        return gym.make(self.env_name)


    def __init__(self, env_name, visualize=False, step_mul=None) -> None:



        print("creating env....")
        self.env_name = env_name
        self.visualize = visualize
        self.step_mul = step_mul

    def run(self, num_episodes=1):


        self.env.settings['visualize'] = self.visualize
        self.env.settings['step_mul'] = self.step_mul

        #episode_rewards = np.zeros((num_episodes, ), dtype=np.int32)
        for ix in range(num_episodes):
            obs = self.env.reset()
            t = 0
            done = False
            while not done:
                action = self.get_action(self.env, obs)


                new_obs, reward, done, _ = self.env.step(action)

                #reward = Tensor([reward])

                memory.push(obs, action, new_obs, reward)
                obs = new_obs

                optimize_model()
                t +=1

            episode_rewards.append(reward)
            #plot_rewards()

        self.env.close()

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
    CUDA=True
    RECURRENT_POLICY=False
    #number of frames to train
    NUM_FRAMES=10e6

    #number of forward steps in A2C (default: 5)
    NUM_STEPS = 5

    NUM_UPDATES = int(NUM_FRAMES) // NUM_STEPS // NUM_AGENTS

    #TODO this should be given by env
    obs_shape = [64, 64]


    envs = [CollectMineralShards_A2C(_ENV_NAME, _VISUALIZE, _STEP_MUL) for i in range(NUM_AGENTS)]
    #jenvs = [gym.make(_ENV_NAME) for i in range(NUM_AGENTS)]

    envs = SubprocVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs)

    obs_shape = envs.observation_space.shape

    obs_shape = (obs_shape[0] * NUM_FRAMES_TO_STACK, *obs_shape[1:])

    if len(envs.observation_space.shape) == 3:
        actor_critic = CNNPolicy(obs_shape[0], envs.action_space, RECURRENT_POLICY)

    action_shape = envs.action_space.shape

    if CUDA:
        actor_critic.cuda()

    optimizer = optim.RMSprop(actor_critic.parameters(), LR, eps=EPS, alpha=ALPHA)
    rollouts = RolloutStorage(NUM_STEPS, NUM_AGENTS, obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(NUM_AGENTS, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if NUM_FRAMES_TO_STACK > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs
    obs = envs.reset()
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([NUM_AGENTS, 1])
    final_rewards = torch.zeros([NUM_AGENTS, 1])

    if CUDA:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()

    for j in range(NUM_UPDATES):
        for step in range(NUM_STEPS):
            # Sample actions

            # Reshape to do in a single forward pass for all steps
            value, action, action_log_prob, states = actor_critic.act(
                Variable(rollouts.observations[step], volatile=True),
                Variable(rollouts.states[step], volatile=True),
                Variable(rollouts.masks[step], volatile=True))

            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if CUDA:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs)
            rollouts.insert(step, current_obs, states.data, action.data, action_log_prob.data, value.data, reward,
                            masks)

        next_value = actor_critic(Variable(rollouts.observations[-1], volatile=True),
                              Variable(rollouts.states[-1], volatile=True),
                              Variable(rollouts.masks[-1], volatile=True))[0].data

    sys.exit()





if __name__ == "__main__":
    main()
