import gym
import ray

env = gym.make("CartPole-v0")
s = env.reset()
a = env.action_space.sample()
# total_episode = 10
# for _ in range(total_episode):
#     total_reward = 0
#     env.reset()
#     done = False
#     while not done:
#         action = env.action_space.sample()
#         s_, r, done, info = env.step(action)
#         total_reward += r
#         env.render()
#     print("total reward: {}".format(total_reward))

