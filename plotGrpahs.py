import matplotlib.pyplot as plt
import numpy as np


rewards_hardcoded = np.load("episodeReward_2.npy")
rewards_random = np.load("episodeReward_1.npy")
rewards_dqn = np.load("episodeReward_0.npy")
def get_mean(rewards):
    r = [0]*100
    for i in range(100,rewards.shape[0]):
        r.append(np.mean(np.array(rewards[i-100:i])))
    return np.array(r)


rewards_hardcoded_mean = get_mean(rewards_hardcoded)
rewards_random_mean = get_mean(rewards_random)
rewards_dqn_mean = get_mean(rewards_dqn)


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