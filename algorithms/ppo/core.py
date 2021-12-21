from gym.spaces import Box, Discrete
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def mlp(hidden_size, activation, output_activation=nn.Identity):
    layers = []
    for i in range(len(hidden_size) - 1):
        act = activation if i < len(hidden_size) - 2 else output_activation
        fc = nn.Linear(hidden_size[i], hidden_size[i + 1])
        fc.weight.data.normal_(0, 0.1)
        layers += [fc, act()]
    return nn.Sequential(*layers)


def discount_cumsum(x, discount):
    """
    折扣累计的计算
    :param x: vector like
       [x0,
        x1,
        x2]
    :param discount: 折扣率
    :return:
    [
        x0 + discount * x1 + discount^2 * x2
        x1 + discount * x2,
        x2
    ]
    """
    # scipy实现:
    # scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    # 手动实现
    res = []
    tmp = x[-1]
    res.append(tmp)
    for i in range(len(x) - 2, -1, -1):
        tmp = x[i] + discount * tmp
        res.append(tmp)
    return res[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None if act is None else self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_size, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        # self.log_std = mlp([obs_dim] + list(hidden_size) + [act_dim], activation)
        self.mu_net = mlp([obs_dim] + list(hidden_size) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Normal的log prob需要


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sized, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sized) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):
    def __init__(self, obs_space, act_space, hidden_size=(64, 64), activation=nn.Tanh):
        super().__init__()
        obs_dim = obs_space.shape[0]

        if isinstance(act_space, Box):
            self.actor = MLPGaussianActor(obs_dim, act_space.shape[0], hidden_size, activation)
        elif isinstance(act_space, Discrete):
            self.actor = MLPCategoricalActor(obs_dim, act_space.n, hidden_size, activation)
        else:
            raise TypeError("action space type error")

        self.critic = MLPCritic(obs_dim, hidden_size, activation)

    def step(self, obs: torch.Tensor):
        # 填充buffer时调用, 不用forward且不用计算梯度, 只是根据obs获取采样的a, logp_a,以及计算得到的v
        # 真正的优化更新参数在读取buffer的时候
        with torch.no_grad():
            pi = self.actor._distribution(obs)
            a = pi.sample()
            logp_a = self.actor._log_prob_from_distribution(pi, a)
            v = self.critic(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()


if __name__ == '__main__':
    x = [1, 2, 3]
    print(discount_cumsum(x, 0.5))
    # x = [1,2,3] discount_rate = 0.5
    # y = [ 1+0.5*2+0.25*3,
    #       2 + 0.5*3,
    #       3,
    #       ]
    # y = [3, 3.5, 2.75]
