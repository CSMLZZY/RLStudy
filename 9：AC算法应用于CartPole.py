import argparse
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorlayer as tl

tl.logging.set_verbosity(tl.logging.DEBUG)

# 设置随机种子，便于重现实验结果
np.random.seed(2)
tf.random.set_seed(2)

# 配置参数解析器
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
print(parser.parse_args())
args = parser.parse_args()

# 设置超参数
OUTPUT_GRAPH = False
TRAIN_EPISODES = 200  # 训练游戏的轮数
TEST_EPISODES = 10  # 测试游戏的轮数
DISPLAY_REWARD_THRESHOLD = 100  # 如果奖励大于此阈值，则渲染游戏环境
MAX_STEPS = 500  # 一轮游戏中的最大运行步数
RENDER = False  # 是否进行渲染
LAMBDA = 0.9  # TD error中的错误折扣率
LR_A = 0.001  # actor的学习率
LR_C = 0.01  # critic的学习率


class Actor(object):

    def __init__(self, n_features, n_actions, lr=0.001):
        # 创建Actor网络
        def get_model(inputs_shape):
            ni = tl.layers.Input(inputs_shape, name='state')
            nn = tl.layers.Dense(n_units=30, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01),
                                 name='hidden')(ni)
            nn = tl.layers.Dense(n_units=10, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01),
                                 name='hidden2')(nn)
            nn = tl.layers.Dense(n_units=n_actions, name='actions')(nn)
            return tl.models.Model(inputs=ni, outputs=nn, name="Actor")

        self.model = get_model([None, n_features])
        # 设置该模型可被训练
        self.model.train()
        self.optimizer = tf.optimizers.Adam(lr)

    # Actor学习
    def learn(self, s, a, td):
        with tf.GradientTape() as tape:
            _logits = self.model(np.array([s]))
            # 带权重更新
            # TD-error就是Actor更新策略时候，带权重更新中的权重值
            _exp_v = tl.rein.cross_entropy_reward_loss(logits=_logits, actions=[a], rewards=td[0])
        grad = tape.gradient(_exp_v, self.model.weights)
        self.optimizer.apply_gradients(zip(grad, self.model.weights))
        return _exp_v

    # 按照分布随机动作。
    def choose_action(self, s):
        _logits = self.model(np.array([s]))
        _probs = tf.nn.softmax(_logits).numpy()
        return tl.rein.choice_action_by_probs(_probs.ravel())

    # 贪婪算法选择动作
    def choose_action_greedy(self, s):
        _logits = self.model(np.array([s]))  # _logits: 动作的概率分布
        _probs = tf.nn.softmax(_logits).numpy()
        return np.argmax(_probs.ravel())

    def save_ckpt(self):
        tl.files.save_npz(self.model.weights, name='model_actor.npz')

    def load_ckpt(self):
        tl.files.load_and_assign_npz(name='model_actor.npz', network=self.model)


class Critic(object):

    def __init__(self, n_features, lr=0.01):
        # 创建Critic网络。
        def get_model(inputs_shape):
            ni = tl.layers.Input(inputs_shape, name='state')
            nn = tl.layers.Dense(n_units=30, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01),
                                 name='hidden')(ni)
            nn = tl.layers.Dense(n_units=5, act=tf.nn.relu, W_init=tf.random_uniform_initializer(0, 0.01),
                                 name='hidden2')(nn)
            nn = tl.layers.Dense(n_units=1, act=None, name='value')(nn)
            return tl.models.Model(inputs=ni, outputs=nn, name="Critic")

        self.model = get_model([1, n_features])
        self.model.train()

        self.optimizer = tf.optimizers.Adam(lr)

    # Critic学习
    def learn(self, s, r, s_):
        v_ = self.model(np.array([s_]))
        with tf.GradientTape() as tape:
            v = self.model(np.array([s]))
            # 计算TD-error
            # TD_error = r + lambd * V(newS) - V(S)
            td_error = r + LAMBDA * v_ - v
            loss = tf.square(td_error)
        grad = tape.gradient(loss, self.model.weights)
        self.optimizer.apply_gradients(zip(grad, self.model.weights))
        return td_error

    def save_ckpt(self):
        tl.files.save_npz(self.model.weights, name='model_critic.npz')

    def load_ckpt(self):
        tl.files.load_and_assign_npz(name='model_critic.npz', network=self.model)


if __name__ == '__main__':
    env = gym.make('CartPole-v1').unwrapped
    # 设置随机种子，使实验结果可重复
    env.seed(2)
    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n

    print("observation dimension: %d" % N_F)  # 4
    print("observation high: %s" % env.observation_space.high)  # [ 2.4 , inf , 0.41887902 , inf]
    print("observation low : %s" % env.observation_space.low)  # [-2.4 , -inf , -0.41887902 , -inf]
    print("num of actions: %d" % N_A)  # 2

    actor = Actor(n_features=N_F, n_actions=N_A, lr=LR_A)
    critic = Critic(n_features=N_F, lr=LR_C)

    # 训练部分
    t0 = time.time()
    if args.train:
        all_episode_reward = []
        for episode in range(TRAIN_EPISODES):
            state = env.reset().astype(np.float32)
            step = 0  # number of step in this episode
            episode_reward = 0  # rewards of all steps
            while True:
                if RENDER:
                    env.render()
                action = actor.choose_action(state)

                state_new, reward, done, info = env.step(action)
                state_new = state_new.astype(np.float32)

                if done:
                    # 希望在濒死状态，可以减去一个较大的reward，让智能体学习如何力挽狂澜。
                    reward = -20

                episode_reward += reward

                try:
                    td_error = critic.learn(state, reward, state_new)  # learn Value-function : gradient = grad[r + lambda * V(s_new) - V(s)]
                    actor.learn(state, action, td_error)  # learn Policy : true_gradient = grad[logPi(s, a) * td_error]
                except KeyboardInterrupt:  # if Ctrl+C at running actor.learn(), then save model, or exit if not at actor.learn()
                    actor.save_ckpt()
                    critic.save_ckpt()

                state = state_new
                step += 1

                if done or step >= MAX_STEPS:
                    break

            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)

            print('Training  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}' \
                  .format(episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0))

            # Early Stopping for quick check
            if step >= MAX_STEPS:
                print("Early Stopping")
                break
        actor.save_ckpt()
        critic.save_ckpt()

        plt.plot(all_episode_reward)
        plt.show()

    # 测试部分
    if args.test:
        actor.load_ckpt()
        critic.load_ckpt()

        for episode in range(TEST_EPISODES):
            episode_time = time.time()
            state = env.reset().astype(np.float32)
            t = 0  # number of step in this episode
            episode_reward = 0
            while True:
                env.render()
                action = actor.choose_action_greedy(state)
                state_new, reward, done, info = env.step(action)
                state_new = state_new.astype(np.float32)
                if done: reward = -20

                episode_reward += reward
                state = state_new
                t += 1

                if done or t >= MAX_STEPS:
                    print('Testing  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}' \
                          .format(episode + 1, TEST_EPISODES, episode_reward, time.time() - t0))
                    break
