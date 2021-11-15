import gym
import numpy as np
import random
import warnings

warnings.filterwarnings('ignore')
env = gym.make("CartPole-v1")


# 环境观察值虽然是四元组，但是其中每一个值都是连续的。这表示游戏中有无数多种环境。那该字典就无法维护，所以需要将其离散化
def discretize(x):
    return tuple((x / np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))


Q = {}
# 0代表向左，1戴代表向右
actions = (0, 1)
# 设置QLearning需要的超参数
alpha = 0.3
gamma = 0.9
epsilon = 0.90


# 获取当前环境下的value值，即向左向右的概率值;如果不存在设置为0
def qvalues(state):
    # dict.get(key, default=None) key -- 字典中要查找的键 default -- 如果指定键的值不存在时，返回该默认值
    return [Q.get((state, a), 0) for a in actions]


# 将向左，向右的可能性压缩到（0，1），且相加为1
def probs(v, eps=1e-4):
    v = v - v.min() + eps
    v = v / v.sum()
    return v


for epoch in range(10000):
    RAll = 0
    obs = env.reset()
    done = False
    while not done:
        s = discretize(obs)
        if random.random() < epsilon:
            # 选择贪婪策略
            v = probs(np.array(qvalues(s)))
            a = random.choices(actions, weights=v)[0]
        else:
            # 选择随机策略
            a = np.random.randint(env.action_space.n)
        obs, rew, done, info = env.step(a)
        RAll += rew
        ns = discretize(obs)
        Q[(s, a)] = Q.get((s, a), 0) + alpha * (rew + gamma * max(qvalues(ns)) - Q.get((s, a), 0))
        s = ns
    print(RAll)
obs = env.reset()
done = False
while not done:
    s = discretize(obs)
    v = probs(np.array(qvalues(s)))
    a = random.choices(actions, weights=v)[0]
    obs, _, done, _ = env.step(a)
env.close()
