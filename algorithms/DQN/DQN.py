import os
import gym
import torch
import random
import numpy as np
from torch import nn
import matplotlib.pyplot as plt



class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size, device):
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.capacity = max_size
        self.ptr = 0
        self.obs_buf = np.zeros(self.combine_shape(max_size, obs_dim), dtype=np.float32)
        self.obs_next_buf = np.zeros(self.combine_shape(max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, 1), dtype=np.int32)
        self.rew_buf = np.zeros((max_size, 1), dtype=np.float32)

    def store(self, s, a, r, s_):
        idx = self.ptr % self.capacity
        self.obs_buf[idx] = s
        self.obs_next_buf[idx] = s_
        self.act_buf[idx] = a
        self.rew_buf[idx] = r
        self.ptr += 1

    def sample_batch(self, batch_size):
        assert batch_size <= self.capacity
        idxs = np.random.randint(0, self.capacity, batch_size) if self.ptr >= self.capacity else \
            np.random.randint(0, self.ptr, batch_size)
        batch_data = [
            torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32).to(self.device),
            torch.as_tensor(self.act_buf[idxs], dtype=torch.long).to(self.device),
            torch.as_tensor(self.rew_buf[idxs], dtype=torch.float32).to(self.device),
            torch.as_tensor(self.obs_next_buf[idxs], dtype=torch.float32).to(self.device)
        ]
        return batch_data

    def combine_shape(self, length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)


class Net(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=None):
        super().__init__()
        self.hidden_size = [16, 64, 256, 32, 16] if not hidden_size else hidden_size
        self.activation = nn.LeakyReLU
        self.model = self.mlp([obs_dim] + self.hidden_size + [act_dim], self.activation)

    def forward(self, x):
        return self.model(x)

    def mlp(self, hidden_size, activation, output_activation=nn.Identity):
        layers = []
        for i in range(len(hidden_size) - 1):
            act = activation if i < len(hidden_size) - 2 else output_activation
            fc = nn.Linear(hidden_size[i], hidden_size[i + 1])
            fc.weight.data.normal_(0, 0.1)
            layers += [fc, act()]
        return nn.Sequential(*layers)


class DQN:
    def __init__(self, env: gym.Env, lr=0.001, init_eps=0.2, gamma=0.9, batch_size=32):
        self.env = env
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n
        self.net = Net(self.obs_dim, self.act_dim).to(self.device).train()
        self.target_net = Net(self.obs_dim, self.act_dim).to(self.device).eval()
        self.loss_fun = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.batch_size = batch_size
        self.init_eps = init_eps
        self.gamma = gamma
        self.lr = lr

    def learn(self, total_episode=5000, update_every_step=1, replay_capacity=2000):
        print("train begin!")
        env = self.env
        replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, replay_capacity, self.device)
        i_step = 0
        is_replay_full = False
        draw_info = []
        for i_episode in range(total_episode):
            episode_reward = 0
            episode_ratio = float(i_episode + 1) / float(total_episode)
            eps = self.init_eps - self.init_eps * episode_ratio
            eps = 0 if eps > 1 else eps
            s = env.reset()
            done = False
            # update target network
            if i_episode % 100 == 0:
                self.target_net.load_state_dict(self.net.state_dict())
            if replay_buffer.ptr >= replay_buffer.capacity:
                is_replay_full = True
            while not done:
                a = self.get_action(s, eps)
                s_, r, done, info = env.step(a)
                # modify the reward
                # x, x_dot, theta, theta_dot = s_
                # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                # r = r1 + r2
                # if done:
                #     r -= 10
                replay_buffer.store(s, a, r, s_)
                episode_reward += r
                i_step += 1
                # buffer不满（其实可以满batch就sample，但buffer不能预先用np.zeros设定好大小了，否则取到0
                if not is_replay_full:
                    continue
                # 固定次数更新一次
                if i_step % update_every_step == 0:
                    batch_data = replay_buffer.sample_batch(self.batch_size)
                    # print(batch_data[1])
                    self.update(batch_data)
                # 关键一步给忘了...
                s = s_
            if is_replay_full:
                info = [i_episode, episode_reward, eps, info["step_num"], info["score"]]
                print("episode {} reward: {:.2f} eps: {:.2f} step_num: {} score: {}".format(*info))
                draw_info.append(info)
        print("train end!")

        # save
        save_path = "./data/models/snakey/episode{}_lr{}_epsilon{}_batchsize{}_gamma{}_{}.pth" \
            .format(total_episode, self.lr, self.init_eps, self.batch_size, self.gamma, self.env.reward_info)
        self.save(save_path)
        self.draw(draw_info, save_path)
        print("model saved")

    def update(self, batch_data):
        b_s, b_a, b_r, b_s_ = batch_data
        q = self.net(b_s).gather(1, b_a)  # 如果buffer中的action不是(maxsize,1)而是(maxsize)的话可以用gather的index参数
        q_target = self.target_net(b_s_).detach()  # 防止更新
        TD_target = b_r + self.gamma * q_target.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_fun(q, TD_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, obs, eps):
        if random.random() < 1 - eps:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            q = self.net(obs_tensor)
            act = torch.max(q, 1)[1].item()
        else:
            act = np.random.randint(0, self.act_dim)
        # 固定返回numpy, 因为
        return np.int32(act)

    def get_action_stable(self, obs):
        if random.random() < 0.9:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            q = self.net(obs_tensor)
            act = torch.max(q, 1)[1].data.cpu().numpy()[0]
        else:
            act = np.random.randint(0, self.act_dim)
        return act

    def save(self, path):
        if os.path.exists(path):
            path = path.replace(".pth", "_.pth")
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def predict(self):
        pass

    def draw(self, draw_info, path):
        episode = np.array([info[0] for info in draw_info], dtype=np.int32)
        reward = np.array([info[1] for info in draw_info], dtype=np.float32)
        epsilon = np.array([info[2] for info in draw_info], dtype=np.float32)
        step_num = np.array([info[3] for info in draw_info], dtype=np.int32)
        score = np.array([info[4] for info in draw_info], dtype=np.int32)
        y_list = [reward, epsilon, step_num, score]
        title_list = ["reward", "epsilon", "step_num", "score"]
        for i in range(4):
            ax = plt.subplot(221 + i)
            ax.set_title(title_list[i])
            ax.plot(episode, y_list[i])
        path = path.replace(".pth", ".png").replace("snakey/", "snakey/imgs/")
        plt.savefig(path)
        plt.show()


if __name__ == '__main__':
    cartpole_env = gym.make("CartPole-v0")
    cartpole_env = cartpole_env.unwrapped
    dqn = DQN(cartpole_env)

    # print(cartpole_env.observation_space.sample())
    # dqn.learn(total_episode=10000)
    # net = Net(cartpole_env.observation_space.shape[0], cartpole_env.action_space.n)
    # print(net)
    # for _ in range(10):
    #     print(random.random())
