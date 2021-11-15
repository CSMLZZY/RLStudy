import argparse
import os
import time
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl
parser=argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train',dest='train',action='store_true',default=True)
parser.add_argument('--test',dest='test',action='store_false')
args=parser.parse_args()
#下面进行超参数的设置
ENV_NAME='Pendulum-v1'
RANDOMSEED=1
LR_A=0.001 #actor网络的学习率
LR_C=0.002 #critic网络的学习率
GAMMA=0.9
TAU=0.01 #软替换（用于更新target网络）,新加入元素的混合比例
MEMORY_CAPACITY=10000
BATCH_SIZE=32 #更新网络时使用的批量大小
MAX_EPISODES=200
MAX_EP_STEPS=200
TEST_PER_EPISODES=10 #每运行10个episode就进行测试
VAR=3 #控制探索
#超参数设置完毕
class DDPG(object):
    def __init__(self,a_dim,s_dim,a_bound):
        #s s_ a done
        self.memory=np.zeros((MEMORY_CAPACITY,s_dim*2+a_dim+1),dtype=np.float32)
        self.pointer=0
        self.a_dim,self.s_dim,self.a_bound=a_dim,s_dim,a_bound
        W_init=tf.random_normal_initializer(mean=0,stddev=0.3)
        b_init=tf.constant_initializer(0.1)
        #搭建actor网络，输入s，输出a
        def get_actor(input_state_shape,name=''):
            inputs=tl.layers.Input(input_state_shape,name='A_input')
            x=tl.layers.Dense(n_units=30,act=tf.nn.relu,W_init=W_init,b_init=b_init,name='A_l1')(inputs)
            #使用tanh将范围限定在[-1,1]
            x=tl.layers.Dense(n_units=a_dim,act=tf.nn.tanh,W_init=W_init,b_init=b_init,name='A_a')(x)
            #进行映射
            x=tl.layers.Lambda(lambda x:np.array(a_bound)*x)(x)
            return tl.models.Model(inputs=inputs,outputs=x,name='Actor'+name)
        #搭建critic网络，输入s、a，输出Q(s,a)
        def get_critic(input_state_shape,input_action_value,name=''):
            s=tl.layers.Input(input_state_shape,name='C_s_input')
            a=tl.layers.Input(input_action_value,name='C_a_input')
            #按照列进行拼接
            x=tl.layers.Concat(1)([s,a])
            x=tl.layers.Dense(n_units=60,act=tf.nn.relu,W_init=W_init,b_init=b_init,name='C_l1')(x)
            x=tl.layers.Dense(n_units=1,W_init=W_init,b_init=b_init,name='C_out')(x)
            return tl.models.Model(inputs=[s,a],outputs=x,name='Critic'+name)
        #建立actor网络与critic网络
        self.actor=get_actor([None,s_dim])
        self.critic=get_critic([None,s_dim],[None,a_dim])
        self.actor.train()
        self.critic.train()
        #初始化target网络的参数，只用于首次赋值，之后不再使用，此函数的作用是为了保持target网络的初始化参数与普通网络的初始化参数保持一致
        def copy_para(from_model,to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i,j in zip(from_model.weights,to_model.weights):
                j.assign(i)
        #建立actor_target网络，并与actor网络的参数一致，不能训练
        self.actor_target=get_actor([None,s_dim],name='_target')
        copy_para(self.actor,self.actor_target)
        self.actor_target.eval()
        #建立critic_target网络，并与critic网络的参数一致，不能训练
        self.critic_target=get_critic([None,s_dim],[None,a_dim],name='_target')
        copy_para(self.critic,self.critic_target)
        self.critic_target.eval()
        #暂时不清楚此层的作用
        self.R=tl.layers.Input([None,1],tf.float32,'r')
        self.ema=tf.train.ExponentialMovingAverage(decay=1-TAU)
        self.actor_opt=tf.keras.optimizers.Adam(LR_A)
        self.critic_opt=tf.keras.optimizers.Adam(LR_C)
    #滑动平均更新
    def ema_update(self):
        paras=self.actor.weights+self.critic.weights
        self.ema.apply(paras)
        for i,j in zip(self.actor_target.weights+self.critic_target.weights,paras):
            i.assign(self.ema.average(j))
    #选择动作，把s带入，输出a
    def choose_action(self,s):
        return self.actor(np.array([s],dtype=np.float32))[0]
    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)  # 随机BATCH_SIZE个随机数
        bt = self.memory[indices, :]  # 根据indices，选取数据bt，相当于随机
        bs = bt[:, :self.s_dim]  # 从bt获得数据s
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  # 从bt获得数据a
        br = bt[:, -self.s_dim - 1:-self.s_dim]  # 从bt获得数据r
        bs_ = bt[:, -self.s_dim:]  # 从bt获得数据s'
        # 更新Critic网络：
        # Critic更新和DQN很像，不过target不是argmax了，是用critic_target计算出来的。
        # br + GAMMA * q_
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)
            q_ = self.critic_target([bs_, a_])
            y = br + GAMMA * q_
            q = self.critic([bs, ba])
            td_error = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic.weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.weights))
        # 更新Actor网络：
        # Actor的目标就是获取最多Q值的。
        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            a_loss = -tf.reduce_mean(q)  # 注意要用负号，是梯度上升！
        a_grads = tape.gradient(a_loss, self.actor.weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.weights))
        self.ema_update()

    # 保存s，a，r，s_
    def store_transition(self, s, a, r, s_):
        # 整理s，s_,方便直接输入网络计算
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)
        # 把s, a, [r], s_横向堆叠
        transition = np.hstack((s, a, [r], s_))
        # pointer记录共有多少数据存储进来过
        # index记录当前最新存储的数据的位置
        # 存储过程是一个循环，当MEMORY_CAPACITY满了以后，index就重新在最底开始了
        index = self.pointer % MEMORY_CAPACITY  # 用新数据替换旧数据
        # 保存transition，即：s, a, [r], s_
        self.memory[index, :] = transition
        self.pointer += 1

    #保存模型
    def save_ckpt(self):
        if not os.path.exists('model'):
            os.makedirs('model')
        tl.files.save_weights_to_hdf5('model/ddpg_actor.hdf5', self.actor)
        tl.files.save_weights_to_hdf5('model/ddpg_actor_target.hdf5', self.actor_target)
        tl.files.save_weights_to_hdf5('model/ddpg_critic.hdf5', self.critic)
        tl.files.save_weights_to_hdf5('model/ddpg_critic_target.hdf5', self.critic_target)
    #读取已保存的模型
    def load_ckpt(self):
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_actor.hdf5', self.actor)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_actor_target.hdf5', self.actor_target)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_critic.hdf5', self.critic)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_critic_target.hdf5', self.critic_target)
if __name__ == '__main__':
    # 初始化环境
    env = gym.make(ENV_NAME)
    """
    创建环境：
    env = gym.make('CartPole-v0')
    返回的这个env其实并非CartPole类本身，而是一个经过包装的环境：
    据说gym的多数环境都用TimeLimit（源码）包装了，以限制Epoch，就是step的次数限制，比如限定为200次。所以小车保持平衡200步后，就会失败(这个现象在DQN的实验中观察到过)。
    用env.unwrapped可以得到原始的类，原始类想step多久就多久，不会200步后失败
    """
    env = env.unwrapped
    # 设置随机种子，为了能够重现（小学期时学到过）
    env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)
    # 定义状态空间，动作空间，动作幅度范围
    s_dim = env.observation_space.shape[0]
    print(env.observation_space.shape)
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    print('s_dim', s_dim)
    print('a_dim', a_dim)
    # 用DDPG算法
    ddpg = DDPG(a_dim, s_dim, a_bound)
    # 训练部分：
    if args.train:
        reward_buffer = []  # 用于记录每个EP的reward，统计变化
        t0 = time.time()  # 统计时间
        for i in range(MAX_EPISODES):
            t1 = time.time()
            s = env.reset()
            ep_reward = 0  # 记录当前EP的reward
            for j in range(MAX_EP_STEPS):
                a = ddpg.choose_action(s)  # 直接用actor估算出a动作
                """
                ddpg.choose_action(s)就是把s整理一下，放入Actor网络，输出action。
                那么如何保证选出来的动作有足够的随机性，能够充分探索环境呢？
                DQN采用的是epsilon-greedy的算法。而DDPG用了正态分布抽样方式。
                """
                #截取，超出的部分就把它置为边界部分
                a = np.clip(np.random.normal(a, VAR), -2, 2)
                # 与环境进行互动
                s_, r, done, info = env.step(a)
                # 保存s，a，r，s_
                ddpg.store_transition(s, a, r / 10, s_)
                # 第一次数据满了，就可以开始学习
                if ddpg.pointer > MEMORY_CAPACITY:
                    ddpg.learn()
                # 输出数据记录
                s = s_
                ep_reward += r  # 记录当前EP的总reward
                if j == MAX_EP_STEPS - 1:
                    print('\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(i+1, MAX_EPISODES, ep_reward,time.time() - t1), end='')
                plt.show()
            # 每隔TEST_PER_EPISODES轮游戏就进行一次测试
            if i and not i % TEST_PER_EPISODES:
                t1 = time.time()
                s = env.reset()
                ep_reward = 0
                for j in range(MAX_EP_STEPS):
                    a = ddpg.choose_action(s)  # 注意，在测试的时候，就不需要用正态分布了，直接用a就可以了。
                    s_, r, done, info = env.step(a)
                    s = s_
                    ep_reward += r
                    if j == MAX_EP_STEPS - 1:
                        print('\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(i+1, MAX_EPISODES, ep_reward,time.time() - t1))
                        reward_buffer.append(ep_reward)
            if reward_buffer:
                plt.ion()
                plt.cla()
                plt.title('DDPG')
                plt.plot(np.array(range(len(reward_buffer))) * TEST_PER_EPISODES, reward_buffer)  # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.ylim(-2000, 0)
                plt.show()
                plt.pause(0.1)
        plt.ioff()
        plt.show()
        print('\nRunning time: ', time.time() - t0)
        ddpg.save_ckpt()
    # test
    ddpg.load_ckpt()
    while True:
        s = env.reset()
        for i in range(MAX_EP_STEPS):
            env.render()
            s, r, done, info = env.step(ddpg.choose_action(s))
            if done:
                break
