import argparse
import time
import gym
import numpy as np
import tensorflow.compat.v1 as tf
import tensorlayer as tl

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()
# 为日志设置开始入口
tl.logging.set_verbosity(tl.logging.DEBUG)
# 设置超参数
lambd = 0.99
e = 0.1
num_episodes = 10000
render = False
running_reward = None


# 实现DQN算法
# 将分类的数字表示，转变为onehot表示
def to_one_hot(i, n_classes=None):
    a = np.zeros(n_classes, 'uint8')
    a[i] = 1
    return a


# 4*4的状态可以用 16 个整数的 one-hot 向量表示
#没有隐藏层，因为只有16个状态，无需隐藏层也可以获得很好的实验效果
def get_model(inputs_shape):
    # W_init和b_init是模型在初始化的时候，随机初始化参数。在此实验中用正态分布，均值0，方差0.01的方式初始化参数
    ni = tl.layers.Input(inputs_shape, name='observation')
    #输出大小为4，对应于四个动作的Q值
    nn = tl.layers.Dense(4, act=None, W_init=tf.random_uniform_initializer(0, 0.01), b_init=None, name='q_a_s')(ni)
    return tl.models.Model(inputs=ni, outputs=nn, name="Q-Network")


# 保存参数
def save_ckpt(model):
    tl.files.save_npz(model.weights, name='dqn_model.npz')


# 加载参数
def load_ckpt(model):
    tl.files.load_and_assign_npz(name='dqn_model.npz', network=model)


if __name__ == '__main__':
    qnetwork = get_model([None, 16])  # 16为state的总数
    qnetwork.train()  # 使用tensorlayer时需要标注该模型是否可以训练
    train_weights = qnetwork.weights  # 模型的参数
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    env = gym.make('FrozenLake-v1')
    # 训练模型
    if args.train:
        t0 = time.time()
        for i in range(num_episodes):
            s = env.reset()
            rAll = 0
            # 最多探索99步
            for j in range(99):
                if render:
                    env.render()
                    # 把state放入network，计算Q值。
                    # 输出，这个状态下，所有动作的Q值，也就是说，是一个[None,4]大小的矩阵
                allQ = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32)).numpy()
                # 在矩阵中找最大的Q值的动作的下标
                a = np.argmax(allQ, 1)
                # e-Greedy：如果小于epsilon，就智能体随机探索。否则，就用最大Q值的动作。
                if np.random.rand(1) < e:
                    # a虽然只包含一个数，但也是一个数组，故需要使用a[0]
                    a[0] = env.action_space.sample()
                # 输入到环境，获得下一步的state，reward，done
                s1, r, d, _ = env.step(a[0])
                # 把new-state 放入，预测下一个state的所有动作的Q值
                Q1 = qnetwork(np.asarray([to_one_hot(s1, 16)], dtype=np.float32)).numpy()
                # 计算target
                # 构建更新target：
                # Q(s,a) <- Q(s,a) + alpha(r + lambd * maxQ(s',a') - Q(s, a))
                maxQ1 = np.max(Q1)  # 下一个状态中最大Q值
                targetQ = allQ  # 用allQ(现在状态的Q值)构建更新的target。因为只有被选择那个动作才会被更新到。
                targetQ[0, a[0]] = r + lambd * maxQ1
                # 利用自动求导 进行更新
                with tf.GradientTape() as tape:
                    _qvalues = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32))  # 把s放入到Q网络，计算_qvalues。
                    # _qvalues和targetQ的差距就是loss。这里衡量的尺子是mse
                    _loss = tl.cost.mean_squared_error(targetQ, _qvalues, is_mean=False)
                # 同梯度带求导对网络参数求导
                grad = tape.gradient(_loss, train_weights)
                # 应用梯度到网络参数求导
                optimizer.apply_gradients(zip(grad, train_weights))
                # 累计reward，并且把s更新为newstate
                rAll += r
                s = s1
                # 更新epsilon，让epsilon随着迭代次数增加而减少。
                # 目的就是智能体越来越少进行“探索”
                if d == True:
                    e = 1. / ((i / 50) + 10)
                    break
            # 这里的running_reward用于记载每一次更新的总和。为了能够更加看清变化，所以大部分是前面的。只有一部分是后面的。
            running_reward = rAll if running_reward is None else running_reward * 0.99 + rAll * 0.01
            # print("Episode [%d/%d] sum reward: %f running reward: %f took: %.5fs " % \
            #     (i, num_episodes, rAll, running_reward, time.time() - episode_time))
            print('Episode: {}/{}  | Episode Reward: {:.4f} | Running Average Reward: {:.4f}  | Running Time: {:.4f}' \
                  .format(i, num_episodes, rAll, running_reward, time.time() - t0))
        save_ckpt(qnetwork)
    #正式进行游戏
    if args.test:
        t0 = time.time()
        #加载训练好的模型
        load_ckpt(qnetwork)
        for i in range(num_episodes):
            episode_time = time.time()
            s = env.reset()
            rAll = 0
            for j in range(99):
                if render:
                    env.render()
                allQ = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32)).numpy()
                a = np.argmax(allQ, 1)  # 不使用训练时所用到的epsilon, 只采取贪心策略
                s1, r, d, _ = env.step(a[0])
                rAll += r
                s = s1
                if d == True:
                    break
            running_reward = rAll if running_reward is None else running_reward * 0.99 + rAll * 0.01
            print('Episode: {}/{}  | Episode Reward: {:.4f} | Running Average Reward: {:.4f}  | Running Time: {:.4f}' \
                  .format(i+1, num_episodes, rAll, running_reward, time.time() - t0))