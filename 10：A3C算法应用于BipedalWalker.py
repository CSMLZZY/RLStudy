import argparse
import multiprocessing
import threading
import time
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl
from tensorlayer.layers import DenseLayer, InputLayer

tfd = tfp.distributions
tl.logging.set_verbosity(tl.logging.DEBUG)
# 为方便实验重现设置随机种子
np.random.seed(2)
tf.random.set_seed(2)
# 配置参数解析器
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()
# 设置超参数
GAME = 'BipedalWalker-v3'  # BipedalWalkerHardcore-v2   BipedalWalker-v2  LunarLanderContinuous-v2
LOG_DIR = './log'  # 保存日志文件的路径
N_WORKERS = multiprocessing.cpu_count()  # 根据 cpu 中的内核数量计算的工人数量
print("n_workers:", N_WORKERS)
MAX_GLOBAL_EP = 800  # 训练进行的游戏的最大轮数
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10  # 在UPDATE_GLOBAL_ITER轮游戏后更新global policy
GAMMA = 0.99  # 奖励折扣因子
ENTROPY_BETA = 0.005  # 熵促进探索的因子
LR_A = 0.00005  # actor的学习率
LR_C = 0.0001  # critic的学习率
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0  # 在训练时会增加, 当 GLOBAL_EP>= MAX_GLOBAL_EP时停止训练


class ACNet(object):
    def __init__(self, scope, globalAC=None):
        self.scope = scope
        self.save_path = './model'
        """
        Glorot 正态分布初始化器，也称为 Xavier 正态分布初始化器。
        它从以 0 为中心，标准差为 stddev = sqrt(2 / (fan_in + fan_out)) 的截断正态分布中抽取样本， 
        其中 fan_in 是权值张量中的输入单位的数量， fan_out 是权值张量中的输出单位的数量。
        """
        w_init = tf.keras.initializers.glorot_normal(seed=None)

        # 输入state，输出actor的分布mu和sigma N(mu,sigma)
        def get_actor(input_shape):
            with tf.name_scope(self.scope):
                ni = tl.layers.Input(input_shape, name='in')
                # relu与relu6的区别见：https://img-blog.csdnimg.cn/20210203160230174.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3ppc3VpbmFfMg==,size_16,color_FFFFFF,t_70
                nn = tl.layers.Dense(n_units=500, act=tf.nn.relu6, W_init=w_init, name='la')(ni)
                nn = tl.layers.Dense(n_units=300, act=tf.nn.relu6, W_init=w_init, name='la2')(nn)
                mu = tl.layers.Dense(n_units=N_A, act=tf.nn.tanh, W_init=w_init, name='mu')(nn)
                sigma = tl.layers.Dense(n_units=N_A, act=tf.nn.softplus, W_init=w_init, name='sigma')(nn)
            return tl.models.Model(inputs=ni, outputs=[mu, sigma], name=scope + '/Actor')

        self.actor = get_actor([None, N_S])
        # 标记为可训练的
        self.actor.train()

        # 输入state，输出V值 注意：不是输出Q值
        def get_critic(input_shape):
            with tf.name_scope(self.scope):
                ni = tl.layers.Input(input_shape, name='in')
                nn = tl.layers.Dense(n_units=500, act=tf.nn.relu6, W_init=w_init, name='lc')(ni)
                nn = tl.layers.Dense(n_units=300, act=tf.nn.relu6, W_init=w_init, name='lc2')(nn)
                v = tl.layers.Dense(n_units=1, W_init=w_init, name='v')(nn)
            return tl.models.Model(inputs=ni, outputs=v, name=scope + '/Critic')

        self.critic = get_critic([None, N_S])
        # 标记为可训练的
        self.critic.train()

    # 更新网络
    @tf.function
    def update_global(self, buffer_s, buffer_a, buffer_v_target, globalAC):
        # 更新全局critic
        with tf.GradientTape() as tape:
            self.v = self.critic(buffer_s)  # V(s)
            self.v_target = buffer_v_target  # V(s')*gamma+r
            td = tf.subtract(self.v_target, self.v, name='TD_error')  # td=v(s')*gamma+r-V(s)
            self.c_loos = tf.reduce_mean(tf.square(td))
        self.c_grads = tape.gradient(self.c_loos, self.critic.weights)  # 注意：求梯度是在本地求的，但更新是global的
        OPT_C.apply_gradients(zip(self.c_grads, globalAC.critic.weights))  # 本地梯度用于全局网络的更新

        # 更新全局actor
        with tf.GradientTape() as tape:
            self.mu, self.sigma = self.actor(buffer_s)  # actor输出mu和sigma
            self.test = self.sigma[0]  # 这里只是为了测试用
            self.mu, self.sigma = self.mu * A_BOUND[1], self.sigma + 1e-5  # mu需要映射到行动空间的范围
            normal_dist = tfd.Normal(self.mu, self.sigma)  # 根据mu和sigma创建正态分布

            self.a_his = buffer_a  # 求action在分布下的概率。float32
            log_prob = normal_dist.log_prob(self.a_his)

            exp_v = log_prob * td  # 带权重更新 td is from the critic part, no gradients for it。

            # 求最大熵
            entropy = normal_dist.entropy()  # 鼓励探索

            self.exp_v = ENTROPY_BETA * entropy + exp_v
            self.a_loss = tf.reduce_mean(-self.exp_v)
        self.a_grads = tape.gradient(self.a_loss, self.actor.weights)
        OPT_A.apply_gradients(zip(self.a_grads, globalAC.actor.weights))  # 本地梯度用于全局网络的更新
        return self.test  # for test purpose

    @tf.function
    def pull_global(self, globalAC):  # run by a local
        # 把全局网络的参数赋值给本地网络
        for l_p, g_p in zip(self.actor.weights, globalAC.actor.weights):
            l_p.assign(g_p)
        for l_p, g_p in zip(self.critic.weights, globalAC.critic.weights):
            l_p.assign(g_p)

    # 选择动作，输入s，输出
    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        self.mu, self.sigma = self.actor(s)

        with tf.name_scope('wrap_a_out'):
            self.mu, self.sigma = self.mu * A_BOUND[1], self.sigma + 1e-5  # sigma增大了少许
        normal_dist = tfd.Normal(self.mu, self.sigma)  # 构建正态分布
        self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *A_BOUND)
        return self.A.numpy()[0]

    def save_ckpt(self):
        tl.files.save_npz(self.actor.weights, name='model_actor.npz')
        tl.files.save_npz(self.critic.weights, name='model_critic.npz')

    def load_ckpt(self):
        tl.files.load_and_assign_npz(name='model_actor.npz', network=self.actor)
        tl.files.load_and_assign_npz(name='model_critic.npz', network=self.critic)


class Worker(object):

    def __init__(self, name, globalAC):
        self.env = gym.make(GAME)  # 创建环境，每个worker都要创建一个环境，是独立的
        self.name = name  # worker的名字
        self.AC = ACNet(name, globalAC)  # AC算法

    def work(self, globalAC):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:  # MAX_GLOBAL_EP最大训练EP
            s = self.env.reset()  # 重置环境
            ep_r = 0  # 统计ep的总reward
            while True:
                # 在训练期间可视化worker0
                if self.name == 'Worker_0' and total_step % 30 == 0:  # worker_0,每30步渲染一次
                    self.env.render()
                s = s.astype('float32')  # double to float
                a = self.AC.choose_action(s)  # 选择动作
                s_, r, done, _info = self.env.step(a)  # 和环境互动

                s_ = s_.astype('float32')  # double to float
                # set robot falls reward to -2 instead of -100
                if r == -100:
                    r = -2  # 把reward-100的时候，改为-2

                ep_r += r  # 保存数据
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                # TD(n)的架构。
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net

                    # 计算最后一步的V(s')
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = self.AC.critic(s_[np.newaxis, :])[0, 0]  # reduce dim from 2 to 0

                    buffer_v_target = []

                    # 计算每个state的V(s')
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)

                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = (
                        np.vstack(buffer_s),
                        np.vstack(buffer_a),
                        np.vstack(buffer_v_target)
                    )

                    # 更新全局网络的参数
                    # update gradients on global network
                    self.AC.update_global(buffer_s, buffer_a, buffer_v_target.astype('float32'), globalAC)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    # update local network from global network
                    self.AC.pull_global(globalAC)

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:  # moving average
                        GLOBAL_RUNNING_R.append(0.95 * GLOBAL_RUNNING_R[-1] + 0.05 * ep_r)
                    print('{}, Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}' \
                          .format(self.name, GLOBAL_EP, MAX_GLOBAL_EP, ep_r, time.time() - t0))
                    GLOBAL_EP += 1
                    break


if __name__ == "__main__":

    env = gym.make(GAME)

    N_S = env.observation_space.shape[0]  # 状态空间
    N_A = env.action_space.shape[0]  # 动作空间
    print("以下为N_S：")
    print(N_S)
    print("以下为N_A：")
    print(N_A)

    A_BOUND = [env.action_space.low, env.action_space.high]  # 动作范围
    A_BOUND[0] = A_BOUND[0].reshape(1, N_A)  # 动作范围形状修改
    A_BOUND[1] = A_BOUND[1].reshape(1, N_A)
    print("以下为A_BOUND")
    print(A_BOUND)
    # 进行训练
    if args.train:
        t0 = time.time()  # 计算时间
        with tf.device("/cpu:0"):  # 以下部分，都在CPU0完成

            OPT_A = tf.optimizers.RMSprop(LR_A, name='RMSPropA')  # 创建Actor的优化器
            OPT_C = tf.optimizers.RMSprop(LR_C, name='RMSPropC')  # 创建Critic的优化器

            GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # 创建全局网络GLOBAL_AC
            workers = []  # workers列表

            # 创建worker
            for i in range(N_WORKERS):
                i_name = 'Worker_%i' % i  # worker name
                workers.append(Worker(i_name, GLOBAL_AC))  # 创建worker，并放在workers列表中，方便统一管理

        COORD = tf.train.Coordinator()  # 创建tensorflow中协调器

        # start TF threading
        worker_threads = []
        for worker in workers:  # 执行每一个worker
            # t = threading.Thread(target=worker.work)
            job = lambda: worker.work(GLOBAL_AC)  # worker要执行的工作。
            t = threading.Thread(target=job)  # 创建一个线程，执行工作
            t.start()  # 开始线程，并执行
            worker_threads.append(t)  # 把线程加入worker_threads中。
        COORD.join(worker_threads)  # 线程由COORD统一管理即可

        # ====画图====
        import matplotlib.pyplot as plt

        plt.plot(GLOBAL_RUNNING_R)
        plt.xlabel('episode')
        plt.ylabel('global running reward')
        plt.show()

        GLOBAL_AC.save_ckpt()

    # 进行测试
    if args.test:
        GLOBAL_AC.load_ckpt()
        while True:
            s = env.reset()
            rall = 0
            while True:
                env.render()
                s = s.astype('float32')  # double to float
                a = GLOBAL_AC.choose_action(s)
                s, r, d, _ = env.step(a)
                rall += r
                if d:
                    print("reward", rall)
                    break
