# TD3的实现包括6个网络：2个Q-net，2个targetq-net，1个policy net，1个target policy net
# TD3 中的 Actor 策略是确定性的，具有高斯探索噪声
import argparse
import math
import random
import time
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from IPython.display import clear_output
import tensorlayer as tl
from tensorlayer.layers import Dense
from tensorlayer.models import Model

tfd = tfp.distributions
Normal = tfd.Normal
tl.logging.set_verbosity(tl.logging.DEBUG)
# 随机种子
random.seed(2)
np.random.seed(2)
tf.random.set_seed(2)
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()
# 设置超参数
ENV = 'Pendulum-v1'
# scale action, [-action_range, action_range]
action_range = 1.0
# 训练总步数
max_frames = 40000
# 测试总步数
test_frames = 300
# 一个episode的最大步数
max_steps = 150
# 更新的batch的大小
batch_size = 64
# 在训练开始时，随机动作采样500
explore_steps = 500
# 单步重复更新
update_itr = 3
# 网络隐藏层的大小
hidden_dim = 32
# Q网络学习率
q_lr = 3e-4
# policy网络学习率
policy_lr = 3e-4
# 更新策略网络和目标网络的延迟步骤
policy_target_update_interval = 3
# 探索的动作噪声范围
explore_noise_scale = 1.0
# 用于评估动作值的动作噪声范围
eval_noise_scale = 0.5
# 奖励的价值范围
reward_scale = 1.0
# 重放缓冲区的大小
replay_buffer_size = 5e5


# 定义重放缓冲区(环形缓冲区)的类
class ReplayBuffer:
    '''
    :state: (state_dim,)
    :action: (action_dim,)
    :reward: (,), 标量
    :next_state: (state_dim,)
    :done: (,), 标量
    '''

    def __init__(self, capacity):
        self.capacity = capacity  # buffer的最大容量
        self.buffer = []  # buffer列表
        self.position = 0  # 当前输入的位置，相当于指针

    def push(self, state, action, reward, next_state, done):
        # 如果buffer的长度小于最大值，即：第一环时，需要先初始化一个空间，这个空间值为None，再给这个空间赋值
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# 学习使用，代码中没有用到，直接修改gym环境的动作输出，把输出归一化。
class NormalizedActions(gym.ActionWrapper):
    # 将动作归一化到合理范围内
    def _action(self, action):
        # 动作空间最小值
        low = self.action_space.low
        # 动作空间最大值
        high = self.action_space.high
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        return action

    # 上一函数的逆过程
    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        return action


# Q网络
class QNetwork(Model):
    # num_inputs相当于state_dim
    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()
        input_dim = num_inputs + num_actions
        # w_init = tf.keras.initializers.glorot_normal(seed=None)
        # 生成均匀分布的随机数
        w_init = tf.random_uniform_initializer(-init_w, init_w)
        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=input_dim, name='q1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='q2')
        # in_channels——上一层的输出大小（这一层的输入大小） n_units——这一层的输出大小
        self.linear3 = Dense(n_units=1, W_init=w_init, in_channels=hidden_dim, name='q3')

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class PolicyNetwork(Model):
    # 用于从状态输入生成非确定性（高斯分布）动作的网络
    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1., init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        # w_init = tf.keras.initializers.glorot_normal(seed=None)
        w_init = tf.random_uniform_initializer(-init_w, init_w)
        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=num_inputs, name='policy1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy2')
        self.linear3 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy3')
        self.output_linear = Dense(n_units=num_actions, W_init=w_init,
                                   b_init=tf.random_uniform_initializer(-init_w, init_w), in_channels=hidden_dim,
                                   name='policy_output')
        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)
        # 单位范围输出
        output = tf.nn.tanh(self.output_linear(x))
        return output

    def evaluate(self, state, eval_noise_scale):
        '''
        生成带有状态的动作以计算梯度
        eval_noise_scale：作为目标策略平滑的技巧，用于生成带有噪声的动作
        '''
        state = state.astype(np.float32)
        # 通过state计算action，注意这里action范围是[-1,1]
        action = self.forward(state)
        # 映射到游戏的action取值范围
        action = self.action_range * action
        # 添加噪声
        # 正态分布
        normal = Normal(0, 1)
        # 对噪声进行上下限裁剪。eval_noise_scale
        eval_noise_clip = 2 * eval_noise_scale
        # noisy和action的shape一致，然后乘以scale
        noise = normal.sample(action.shape) * eval_noise_scale
        # 对noisy进行剪切
        noise = tf.clip_by_value(noise, -eval_noise_clip, eval_noise_clip)
        # 给动作添加噪声
        action = action + noise
        return action

    # 输入state，输出action
    def get_action(self, state, explore_noise_scale):
        # 生成带有状态的动作以与环境交互
        # 这里的forward函数，就是输入state，然后通过state输出action。只不过形式不一样而已。最后的激活函数式tanh，所以范围是[-1, 1]
        action = self.forward([state])
        # 获得的action变成矩阵。
        action = action.numpy()[0]
        # 添加噪声
        # 正态分布
        normal = Normal(0, 1)
        # 在正态分布中抽样一个和action一样shape的数据，然后乘以scale
        noise = normal.sample(action.shape) * explore_noise_scale
        # action乘以动作的范围，加上noise
        action = self.action_range * action + noise
        return action.numpy()

    def sample_action(self, ):
        # 生成用于探索的随机动作
        a = tf.random.uniform([self.num_actions], -1, 1)
        return self.action_range * a.numpy()


class TD3_Trainer():
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_dim, action_range, policy_target_update_interval=1,
                 q_lr=3e-4, policy_lr=3e-4):
        self.replay_buffer = replay_buffer
        # 初始化用到的所有网络
        # 用两个Q_net来估算，同时也有两个对应的target_q_net
        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)
        # 初始化target网络的权重
        # 把net赋值给target_network
        self.target_q_net1 = self.target_init(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_init(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_init(self.policy_net, self.target_policy_net)
        # 更新次数
        self.update_cnt = 0
        # 策略网络更新频率
        self.policy_target_update_interval = policy_target_update_interval
        self.q_optimizer1 = tf.keras.optimizers.Adam(q_lr)
        self.q_optimizer2 = tf.keras.optimizers.Adam(q_lr)
        self.policy_optimizer = tf.keras.optimizers.Adam(policy_lr)

    # 在target网络初始化时进行硬更新
    def target_init(self, net, target_net):
        for target_param, param in zip(target_net.weights, net.weights):
            target_param.assign(param)
        return target_net

    # 在更新的时候进行软更新
    def target_soft_update(self, net, target_net, soft_tau):
        for target_param, param in zip(target_net.weights, net.weights):
            # 原来参数占比 + 目前参数占比
            target_param.assign(target_param * (1.0 - soft_tau) + param * soft_tau)
        return target_net

    # 进行所有网络的更新
    def update(self, batch_size, eval_noise_scale, reward_scale=10., gamma=0.9, soft_tau=1e-2):
        # 计算更新次数
        self.update_cnt += 1
        # 从回放缓冲区中随机取出batch_size条数据
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        """
        np.newaxis的作用是增加一个维度。
        对于[: , np.newaxis] 和 [np.newaxis，：]
        是在np.newaxis这里增加1维。
        这样改变维度的作用往往是将一维的数据转变成一个矩阵，与代码后面的权重矩阵进行相乘， 否则单单的数据是不能这样相乘的。
        """
        # 调整形状，方便输入网络
        reward = reward[:, np.newaxis]
        done = done[:, np.newaxis]
        # 输入s',从target_policy_net计算a'。注意计算得出的action已经添加了noisy
        new_next_action = self.target_policy_net.evaluate(next_state, eval_noise_scale=eval_noise_scale)
        # 归一化reward.(有正有负)
        reward = reward_scale * (reward - np.mean(reward, axis=0)) / (np.std(reward, axis=0) + 1e-6)

        # Training Q Function
        # 把s'和a'堆叠在一起，一起输入到target_q_net。
        # 有两个qnet，取最小值
        target_q_input = tf.concat([next_state, new_next_action], 1)
        target_q_min = tf.minimum(self.target_q_net1(target_q_input), self.target_q_net2(target_q_input))

        # 计算target_q的值，用于更新q_net
        # 之前有把done从布尔变量改为int，就是为了这里能够直接计算。
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward

        q_input = tf.concat([state, action], 1)
        # 更新q_net1
        # 和DQN是一样的
        with tf.GradientTape() as q1_tape:
            predicted_q_value1 = self.q_net1(q_input)
            q_value_loss1 = tf.reduce_mean(tf.square(predicted_q_value1 - target_q_value))
        q1_grad = q1_tape.gradient(q_value_loss1, self.q_net1.weights)
        self.q_optimizer1.apply_gradients(zip(q1_grad, self.q_net1.weights))

        # 更新q_net2
        with tf.GradientTape() as q2_tape:
            predicted_q_value2 = self.q_net2(q_input)
            q_value_loss2 = tf.reduce_mean(tf.square(predicted_q_value2 - target_q_value))
        q2_grad = q2_tape.gradient(q_value_loss2, self.q_net2.weights)
        self.q_optimizer2.apply_gradients(zip(q2_grad, self.q_net2.weights))

        # Training Policy Function
        # policy_net不是经常更新的，而是qnet更新一定次数后，policy_net才更新一次
        if self.update_cnt % self.policy_target_update_interval == 0:
            # 更新policy_net
            with tf.GradientTape() as p_tape:
                # 计算 action = Policy(s)，注意这里是没有noise的，确定性的梯度策略
                new_action = self.policy_net.evaluate(state, eval_noise_scale=0.0)
                # 叠加state和action
                new_q_input = tf.concat([state, new_action], 1)
                # predicted_new_q_value = tf.minimum(self.q_net1(new_q_input),self.q_net2(new_q_input))
                predicted_new_q_value = self.q_net1(new_q_input)
                # 注意：此处是梯度上升！
                policy_loss = -tf.reduce_mean(predicted_new_q_value)
            p_grad = p_tape.gradient(policy_loss, self.policy_net.weights)
            self.policy_optimizer.apply_gradients(zip(p_grad, self.policy_net.weights))

            # 软更新三个target网络
            self.target_q_net1 = self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2 = self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net = self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

    def save_weights(self):
        tl.files.save_npz(self.q_net1.weights, name='model_q_net1.npz')
        tl.files.save_npz(self.q_net2.weights, name='model_q_net2.npz')
        tl.files.save_npz(self.target_q_net1.weights, name='model_target_q_net1.npz')
        tl.files.save_npz(self.target_q_net2.weights, name='model_target_q_net2.npz')
        tl.files.save_npz(self.policy_net.weights, name='model_policy_net.npz')
        tl.files.save_npz(self.target_policy_net.weights, name='model_target_policy_net.npz')

    def load_weights(self):
        tl.files.load_and_assign_npz(name='model_q_net1.npz', network=self.q_net1)
        tl.files.load_and_assign_npz(name='model_q_net2.npz', network=self.q_net2)
        tl.files.load_and_assign_npz(name='model_target_q_net1.npz', network=self.target_q_net1)
        tl.files.load_and_assign_npz(name='model_target_q_net2.npz', network=self.target_q_net2)
        tl.files.load_and_assign_npz(name='model_policy_net.npz', network=self.policy_net)
        tl.files.load_and_assign_npz(name='model_target_policy_net.npz', network=self.target_policy_net)


# 绘制图像
def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.savefig('td3.png')
    # plt.show


if __name__ == '__main__':
    # 初始化环境
    # env = NormalizedActions(gym.make(ENV))
    env = gym.make(ENV).unwrapped
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    # 初始化回放缓冲区
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # 初始化trainer
    td3_trainer = TD3_Trainer(replay_buffer, state_dim, action_dim, hidden_dim=hidden_dim,
                              policy_target_update_interval=policy_target_update_interval, \
                              action_range=action_range, q_lr=q_lr, policy_lr=policy_lr)

    # 设置可被训练
    td3_trainer.q_net1.train()
    td3_trainer.q_net2.train()
    td3_trainer.target_q_net1.train()
    td3_trainer.target_q_net2.train()
    td3_trainer.policy_net.train()
    td3_trainer.target_policy_net.train()

    # 设置网络可训练
    td3_trainer.q_net1.train()
    td3_trainer.q_net2.train()
    td3_trainer.target_q_net1.train()
    td3_trainer.target_q_net2.train()
    td3_trainer.policy_net.train()
    td3_trainer.target_policy_net.train()

    # 训练
    if args.train:
        frame_idx = 0  # 总步数
        rewards = []  # 记录每个EP的总reward
        t0 = time.time()
        while frame_idx < max_frames:  # 小于最大步数，就继续训练
            state = env.reset()  # 初始化state
            state = state.astype(np.float32)  # 整理state的类型
            episode_reward = 0
            if frame_idx < 1:  # 第一次的时候，要进行初始化trainer
                print('intialize')
                # 这里需要额外的调用才能使内部函数能够使用模型转发
                _ = td3_trainer.policy_net([state])
                _ = td3_trainer.target_policy_net([state])

            # 开始训练
            for step in range(max_steps):
                if frame_idx > explore_steps:  # 如果小于500步，就随机，如果大于就用get-action
                    action = td3_trainer.policy_net.get_action(state, explore_noise_scale=1.0)  # 带有noisy的action
                else:
                    action = td3_trainer.policy_net.sample_action()

                # 与环境进行交互
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.astype(np.float32)
                env.render()
                done = 1 if done == True else 0

                # 记录数据在replay_buffer
                replay_buffer.push(state, action, reward, next_state, done)

                # 赋值state，累计总reward，步数
                state = next_state
                episode_reward += reward
                frame_idx += 1

                # 如果数据超过一个batch_size的大小，那么就开始更新
                if len(replay_buffer) > batch_size:
                    for i in range(update_itr):  # 注意：这里更新可以更新多次！
                        td3_trainer.update(batch_size, eval_noise_scale=0.5, reward_scale=1.)

                if frame_idx % 500 == 0:
                    plot(frame_idx, rewards)

                if done:
                    break
            episode = int(frame_idx / max_steps)  # current episode
            all_episodes = int(max_frames / max_steps)  # total episodes
            print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}' \
                  .format(episode, all_episodes, episode_reward, time.time() - t0))
            rewards.append(episode_reward)
        td3_trainer.save_weights()

    if args.test:
        frame_idx = 0
        rewards = []
        t0 = time.time()

        # 加载训练好的网络
        td3_trainer.load_weights()

        while frame_idx < test_frames:
            state = env.reset()
            state = state.astype(np.float32)
            episode_reward = 0
            if frame_idx < 1:
                print('intialize')
                _ = td3_trainer.policy_net([state])
                _ = td3_trainer.target_policy_net([state])

            for step in range(max_steps):
                action = td3_trainer.policy_net.get_action(state, explore_noise_scale=1.0)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.astype(np.float32)
                env.render()
                done = 1 if done == True else 0

                state = next_state
                episode_reward += reward
                frame_idx += 1

                # if frame_idx % 50 == 0:
                #     plot(frame_idx, rewards)

                if done:
                    break
            episode = int(frame_idx / max_steps)
            all_episodes = int(test_frames / max_steps)
            print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}' \
                  .format(episode, all_episodes, episode_reward, time.time() - t0))
            rewards.append(episode_reward)
