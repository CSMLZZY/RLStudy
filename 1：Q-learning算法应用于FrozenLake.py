import numpy as np
import gym
import time

# FrozenLake-v1是一个4*4的网络格子，每个格子可以是起始块，目标块、冻结块或者危险块。
# 我们的目标是让智能体学习如何从开始块如何行动到目标块上，而不是移动到危险块上。
# 智能体可以选择向上、向下、向左或者向右移动，同时游戏中还有可能吹来一阵风，将智能体吹到任意的方块上。
env = gym.make('FrozenLake-v1')
render = False
running_reward = None
Q = np.zeros([env.observation_space.n, env.action_space.n])
print(Q.shape)
# 设置超参数
lr = 0.85
lamda = 0.99
num_episodes = 10000
rList = []
for i in range(num_episodes):
    # 记录运行时间
    episode_time = time.time()
    # 重置初始状态
    s = env.reset()
    # 记录此次游戏的总奖励
    rAll = 0
    # 开始QL算法（设定一次游戏最多进行99步）
    for j in range(99):
        if render:
            env.render()
        # 从Q表格中，找到当前状态S最大Q值，并在Q值上加上噪音。
        # 然后找到最大的Q+
        # np.random.randn(1, env.action_space.n)就是制造出来的噪音，我们希望噪音随着迭代的进行，会越来越小。因此乘以(1. / (i + 1))。当i越来越大的时候，噪音就越来越小了。
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1.0 / (i + 1)))
        # 与环境互动，把动作放到env.step()函数，并返回下一状态S1，奖励，done，info
        s1, r, d, _ = env.step(a)
        Q[s, a] = Q[s, a] + lr * (r + lamda * np.max(Q[s1, :]) - Q[s, a])
        rAll += r
        s = s1
        if d == True:
            break
    rList.append(rAll)
    # 每一次迭代获得的总收获rAll,会以0.01的份额加入到running_reward。(原代码这里rAll用了r，个人认为是rAll更合适)
    running_reward = rAll if running_reward is None else running_reward * 0.99 + rAll * 0.01
    print("Episode [%d/%d] sum reward: %f running reward: %f took: %.5fs " % (
    i, num_episodes, rAll, running_reward, time.time() - episode_time))
print("Final Q-table Values:/n %s" % Q)
