from gridworld import *
from random import random
from gym import Env
import gym


class SarsaAgent():
    def __init__(self, env: Env):
        self.env = env
        self.Q = {}
        self._init_agent()
        self.state = None

    def _init_agent(self):
        self.state = self.env.reset()
        s_name = self._get_state_name(self.state)
        self._assert_state_in_Q(s_name, randomized=True)

    def _get_state_name(self, state):
        return str(state)

    def _is_state_in_Q(self, s):
        return self.Q.get(s) is not None

    def _init_state_value(self, s_name, randomized=True):
        if not self._is_state_in_Q(s_name):
            self.Q[s_name] = {}
            for action in range(self.env.action_space.n):
                default_v = random() / 10 if randomized else 0.0
                self.Q[s_name][action] = default_v

    def _assert_state_in_Q(self, s, randomized=True):
        if not self._is_state_in_Q(s):
            self._init_state_value(s, randomized)

    def _get_Q(self, s, a):
        self._assert_state_in_Q(s, randomized=True)
        return self.Q[s][a]

    def _set_Q(self, s, a, value):
        self._assert_state_in_Q(s, randomized=True)
        self.Q[s][a] = value

    def perform_policy(self, s, episode_num, use_epsilon):
        return self.SARSA(s, episode_num, use_epsilon)

    def SARSA(self, s, episode_num, use_epsilon):
        epsilon = 1.00 / (episode_num + 1)
        Q_s = self.Q[s]
        rand_value = random()
        action = None
        if use_epsilon and rand_value < epsilon:
            action = self.env.action_space.sample()
        else:
            str_act = max(Q_s, key=Q_s.get)
            action = int(str_act)
        return action

    def act(self, a):
        return self.env.step(a)

    def learning(self, gamma, alpha, max_episode_num):
        total_time, time_in_episode, num_episode = 0, 0, 0
        while num_episode < max_episode_num:
            self.state = self.env.reset()
            s0 = self._get_state_name(self.state)
            self.env.render()
            a0 = self.perform_policy(s0, num_episode, use_epsilon=True)
            time_in_episode = 0
            is_done = False

            while not is_done:
                s1, r1, is_done, info = self.act(a0)
                self.env.render()
                s1 = self._get_state_name(s1)
                self._assert_state_in_Q(s1, randomized=True)
                # 评估策略的action，如果use_epsilon 改为 False，则变成 Q-learning
                a1 = self.perform_policy(s1, num_episode, use_epsilon=True)
                old_q = self._get_Q(s0, a0)
                td_target = r1 + gamma * self._get_Q(s1, a1)
                new_q = old_q + alpha * (td_target - old_q)
                self._set_Q(s0, a0, new_q)

                if num_episode == max_episode_num:
                    print("t:{0:>2}: s:{1}, a:{2:2}, s1:{3}". \
                          format(time_in_episode, s0, a0, s1))

                # 有点疑惑，按理说SARSA算法中，更新策略通过ep-greedy拿到的a1不一定会是下一次要执行的action
                # 此处的a0是否要重新通过policyPerform得到？
                # 此外如果用Q-learning，则必须用policyPerform重新得到a0，因为无论是SARSA还是Q-learning的行为策略都是ep-greedy
                # 所以综合考虑还是用a0 = PolicyPerform(s1, num_episode, use_epsilon=True)好点
                s0, a0, = s1, a1
                time_in_episode += 1
            print("Episode {0} takes {1} steps.".format(
                num_episode, time_in_episode))
            total_time += time_in_episode
            num_episode += 1


if __name__ == '__main__':
    env = CliffWalk()
    directory = "./monitor"
    env = gym.wrappers.Monitor(env, directory, force=True)
    agent = SarsaAgent(env)
    env.reset()
    print("learning...")
    agent.learning(
        gamma=0.9,
        alpha=0.1,
        max_episode_num=500
    )