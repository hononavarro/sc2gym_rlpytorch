import matplotlib.pyplot as plt
import numpy as np


rewards_hardcoded = np.load("episodeReward_2.npy")
rewards_random = np.load("episodeReward_1.npy")
rewards_dqn = np.load("episodeReward_0.npy")
rewards_doubleduelingdqn = np.load("episodeReward_0DoubleDuelingQN.npy")
reward_dqn_2agents = np.load("episodeReward_DQN2agents0.npy")

def get_mean(rewards):
    r = [0]*100
    for i in range(100,rewards.shape[0]):
        r.append(np.mean(np.array(rewards[i-100:i])))
    return np.array(r)


rewards_hardcoded_mean = get_mean(rewards_hardcoded)
rewards_random_mean = get_mean(rewards_random)
rewards_dqn_mean = get_mean(rewards_dqn)
rewards_doubleduelingdqn_mean = get_mean(rewards_doubleduelingdqn)
reward_dqn_2agents_mean = get_mean(reward_dqn_2agents)



plt.figure(1)
plt.clf()
plt.title('Score(Hard Coded Path)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.plot(rewards_hardcoded)
plt.plot(rewards_hardcoded_mean)
plt.legend(['Episode reward','100 Episode avg'])
plt.show()



plt.figure(2)
plt.clf()
plt.title('Score(Random)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.plot(rewards_random)
plt.plot(rewards_random_mean)
plt.legend(['Episode reward','100 Episode avg'])
plt.show()


plt.figure(3)
plt.clf()
plt.title('Score(DQN)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.plot(rewards_dqn)
plt.plot(rewards_dqn_mean)
plt.legend(['Episode reward','100 Episode avg'])
plt.show()


plt.figure(4)
plt.clf()
plt.title('Score(DoubleDuelingDQN)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.plot(rewards_doubleduelingdqn)
plt.plot(rewards_doubleduelingdqn_mean)
plt.legend(['Episode reward','100 Episode avg'])
plt.show()


plt.figure(5)
plt.clf()
plt.title('Score(DQN_2agents)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.plot(reward_dqn_2agents)
plt.plot(reward_dqn_2agents_mean)
plt.legend(['Episode reward','100 Episode avg'])
plt.show()