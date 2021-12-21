import time
from .core import *
from pathlib import Path
from torch.optim import Adam
import matplotlib.pyplot as plt


class PPOBuffer(object):

    def __init__(self, obs_dim, act_dim, max_size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(self.combined_shape(max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(self.combined_shape(max_size, act_dim), dtype=np.float32)

        # GAE-lambda计算过后优势函数的值
        self.adv_buf = np.zeros(max_size, dtype=np.float32)
        # GAE-lambda中,lambda=1的情况,即只有gamma的累积值,用于代替reward与网络输出的v计算loss
        self.ret_buf = np.zeros(max_size, dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.val_buf = np.zeros(max_size, dtype=np.float32)
        self.logp_buf = np.zeros(max_size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        # ptr纪录当前指针, start idx临时记录path的头指针,
        self.ptr, self.path_start_idx, self.capacity = 0, 0, max_size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.capacity
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        在行为轨迹的终点或者epoch结束时调用. 调用时会回顾轨迹开始的地方, 根据reward和value的buffer,
        利用GAE-lambda来计算和评估整个轨迹的优势. 同时也计算每个状态下的rewards-to-go(即lambda=1情况下的GAE计算)
        作为更新Critical(V网络)的目标值.

        :param last_val: 如果是轨迹终点(即游戏结束)则此参数为0. 否则必须是最终state的V网络评估的值V(s_T).
        :return:
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.rew_buf[path_slice], last_val)

        # 计算优势函数delta_t(t=[0,T]); delta_0=deltas[0], delta_1=deltas[1],...
        # delta
        # delta_t = r_t + gamma*V(S_(t+1)) - V(S_t)
        # delta_{t+1} = r_{t+1} + gamma*V(S_(t+2)) - V(S_{t+1})
        # Adv
        # adv_1 = delta_t = r_t + gamma*V(S_(t+1)) - V(S_t)
        # adv_2 = delta_t + gamma*delta_{t+1} =     r_t + gamma*V(S_(t+1)) - V(S_t)
        #                               + gamma*r_{t+1} + gamma^2*V(S_(t+2)) - gamma*V(S_{t+1})
        #                                     = r_t + gamma*r_{t+1} + gamma^2*V(S_(t+2)) - V(S_t)
        # adv_k = - V(S_t) + \sum_{t=0}^{k-1} {gamma^k * r_{t+k-1}} + gamma^k * V(s_{t+k})
        # GAE-lam = (1-lam)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        # 计算GAE-Lambda
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # 计算rewards-to-go作为价值网络V的目标值. 记得[:-1], 其中-1是单个r_t
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.capacity
        # get之后buf会被清空
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def combined_shape(self, length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)


class PPO(object):
    def __init__(self, env, step_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2,
                 actor_lr=3e-4, critic_lr=1e-3, train_pi_iters=80, train_v_iters=80,
                 lam=0.97, max_ep_len=1000, target_kl=0.01, save_freq=10):
        self.env = env
        self.obs_dim = env.observation_space.shape
        self.act_dim = env.action_space.shape
        print(env.action_space)
        print(env.action_space.shape)
        print(env.observation_space)
        print(env.observation_space.shape)
        exit()
        self.ac = MLPActorCritic(env.observation_space, env.action_space)

        self.step_per_epoch = step_per_epoch
        self.epochs = epochs
        # 折扣损失
        self.gamma = gamma
        # GAE-lambda中的lambda,
        self.lam = lam
        # PPO1截断率
        self.clip_ratio = clip_ratio
        # pi和v网络学习率
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        # pi和v网络各训练的次数
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters

        # 网络参数优化器
        self.pi_optimizer = Adam(self.ac.actor.parameters(), lr=self.actor_lr)
        self.v_optimizer = Adam(self.ac.critic.parameters(), lr=self.critic_lr)
        # 一轮(trajectory / episode)游戏最长的step次数,超过次数就算done为True也结束
        # 注意episode和epoch的区别
        # 一个epoch包含"step_per_epoch"总步数, 一个epoch能包含多个episode,每个episode最多执行"max_ep_len"步
        self.max_ep_len = max_ep_len
        # 当新老策略的KL散度大于此值时, 执行early stopping,跳过policy pi的更新
        self.target_kl = target_kl
        self.save_freq = save_freq

        self.buffer = PPOBuffer(self.obs_dim, self.act_dim, self.step_per_epoch, self.gamma, self.lam)

    def learn(self, model_save_root):
        ac = self.ac
        env = self.env
        buffer = self.buffer

        draw_info = {
            "trajectory_info": [],
            "update_info": []
        }

        obs, ep_ret, trajectory_idx = env.reset(), 0, 0
        for epoch in range(self.epochs):
            # 一个epoch可能有多个finished path
            for t in range(self.step_per_epoch):
                act, v, logp_a = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                obs_next, r, done, info = env.step(act)

                ep_ret += r

                buffer.store(obs, act, r, v, logp_a)

                obs = obs_next

                time_out = t == self.step_per_epoch - 1
                if done or time_out:
                    # epoch自然完结, 提前计算得到下一个obs的v并且计算GAE-Lambda
                    if time_out:
                        # print('Warning: trajectory cut off by epoch at %d steps.' % t, flush=True)
                        _, v, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                    # done
                    else:
                        v = 0
                    buffer.finish_path(v)
                    # print("trajectory done at: {} steps, total reward: {}".format(t, ep_ret))
                    draw_info["trajectory_info"].append([trajectory_idx, ep_ret, info["step_num"], info["score"]])
                    trajectory_idx += 1
                    obs, ep_ret = env.reset(), 0
            update_info = self.update(buffer)
            draw_info["update_info"].append((epoch, update_info))
            print("epoch: {} loss pi: {} loss v: {}".format(epoch, update_info[0], update_info[2]))
            # last epoch
            if epoch == self.epochs - 1:
                model_save_root = Path(model_save_root)
                if not model_save_root.exists():
                    model_save_root.mkdir(exist_ok=True)
                prefix = "epoch{}_dead{}_apple{}".format(epoch, self.env.reward_dead, self.env.reward_apple)
                model_save_path = model_save_root.joinpath(prefix + ".pt")
                plt_save_root = model_save_root.joinpath("imgs")
                self.save_model(model_save_path)
                self.draw(draw_info, plt_save_root, prefix)

    def update(self, buf: PPOBuffer):
        # buf get 以后buf将被[清空(划掉)]被重置, 从0开始计. 但是可能会混杂上个epoch遗留的数据残渣?
        data = buf.get()

        # 需要先计算一个pi loss old 和 v loss old, 与新的pi和v的loss进行对比, 存入log
        loss_pi_old, pi_info_old = self.compute_loss_pi(data)
        loss_pi_old = loss_pi_old.item()
        loss_v_old = self.compute_loss_v(data).item()

        pi_optimizer = self.pi_optimizer
        v_optimizer = self.v_optimizer

        # 先固定V网络, 训练pi网络
        for i in range(self.train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            # KL散度过大, 不满足置信区间, 停止训练
            if pi_info["kl"] > 1.5 * self.target_kl:
                print("Early stopping at step {} due to reaching max kl.".format(i))
                break
            loss_pi.backward()
            pi_optimizer.step()

        # 训练V网络
        for i in range(self.train_v_iters):
            v_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            v_optimizer.step()

        # 打印信息
        # print("""update end.
        # loss pi old: {} loss pi end: {} diff loss pi {}
        # loss V old: {} loss V end: {} diff loss V {}
        # kl: {}         entropy: {}    clip fraction: {}
        # """.format(loss_pi_old, loss_pi.item(), loss_pi.item() - loss_pi_old,
        #            loss_v_old, loss_v.item(), loss_v.item() - loss_v_old,
        #            pi_info["kl"], pi_info["ent"], pi_info["cf"]))
        info = [loss_pi.item(), loss_pi.item() - loss_pi_old,
                loss_v.item(), loss_v.item() - loss_v_old,
                pi_info["kl"], pi_info["ent"], pi_info["cf"]]
        return info

    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        # policy loss
        # pi是一个分布 like Categorical(logits) or Normal(mu, std); logp 是选中此act的概率p取log
        pi, logp = self.ac.actor(obs, act)
        ratio = torch.exp(logp - logp_old)
        # 截断的ratio和adv. 说明详见Notion笔记
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        # 取min用于区分A>0和A<0的情况. 取负号是因为梯度下降, 原公式为奖励函数用梯度上升.
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # 额外信息 kl散度用于判断俩分布的距离, 用于Early Stopping,防止分布差距过大.
        approx_kl = (logp_old - logp).mean().item()
        entropy = pi.entropy().mean().item()
        # 判断ratio在clip范围内还是外. gt= greater than; lg = less than
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        # 在clip范围内的比例(fraction)?
        clip_frac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=entropy, cf=clip_frac)

        return loss_pi, pi_info

    def compute_loss_v(self, data):
        # obs 和 累计折扣的奖励
        obs, ret = data["obs"], data["ret"]
        # V网络通过obs预测的v值,和trajectory实际折扣奖励的MSE
        return ((self.ac.critic(obs) - ret) ** 2).mean()

    def draw(self, draw_info, save_root, prefix):
        trajectory_info = draw_info["trajectory_info"]
        update_info = draw_info["update_info"]

        # trajectory info
        episode = np.array([info[0] for info in trajectory_info], dtype=np.int32)
        reward = np.array([info[1] for info in trajectory_info], dtype=np.float32)
        step_num = np.array([info[2] for info in trajectory_info], dtype=np.int32)
        score = np.array([info[3] for info in trajectory_info], dtype=np.int32)

        # update info
        epoch = np.array([info[0] for info in update_info], dtype=np.float32)
        loss_pi = np.array([info[1][0] for info in update_info], dtype=np.float32)
        loss_v = np.array([info[1][2] for info in update_info], dtype=np.float32)
        y_list = [reward, step_num, score, loss_pi, loss_v]
        title_list = ["reward", "step_num", "score", "loss_pi", "loss_v"]
        for i in range(3):
            ax = plt.subplot(321 + i)
            ax.set_title(title_list[i])
            ax.plot(episode, y_list[i])
        for i in range(2):
            ax = plt.subplot(321 + i + 3)
            ax.set_title(title_list[i + 3])
            ax.plot(epoch, y_list[i + 3])

        if not save_root.exists():
            save_root.mkdir(exist_ok=True)
        save_path = save_root.joinpath(prefix + ".png")
        plt.savefig(save_path)
        plt.show()

    def save_model(self, path):
        print(path)
        torch.save(self.ac.state_dict(), path)

    def load_model(self, path):
        self.ac.load_state_dict(torch.load(path))

    def get_action(self, obs, eps=0):
        act, _, _ = self.ac.step(torch.as_tensor(obs, dtype=torch.float32))
        return act
