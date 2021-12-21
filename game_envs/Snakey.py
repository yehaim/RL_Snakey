import os
import pygame
import random
import numpy as np
from gym import Env
from refers.gridworld import *



class Snake(object):
    dirs = [
        np.array([-1, 0]),  # left
        np.array([1, 0]),  # right
        np.array([0, -1]),  # up
        np.array([0, 1])  # down
    ]

    def __init__(self, grid_w, grid_h, init_len=4):
        self.grid_w = grid_w
        self.grid_h = grid_h

        self.init_len = init_len  # 初始化贪吃蛇的长度, 长度不包括头

        self.pos = None  # 初始化贪吃蛇头部position坐标，位于显示区域的中心
        self.dir = None  # 初始化贪吃蛇的运动方向
        self.prev_pos = None  # 建立一个position数据用于存贮贪吃蛇历史轨迹，这些轨迹也是贪吃蛇的身子
        self.len = 0
        self.reset()

    def move(self, direction=None):
        """如果不传参,则按原来方向走"""
        # 如果输入方向和原方向相反,则默认按原方向的方向走, 否则不变
        if direction is not None and not ((direction + self.dir) == np.array([0, 0])).all():
            self.dir = direction
        self.pos += self.dir
        self.prev_pos.append(self.pos.copy())
        self.prev_pos = self.prev_pos[-self.len - 1:]

    def check_dead(self, pos):  # 判断贪吃蛇的头部是否到达边界或则碰到自己的身体。
        if pos[0] < 0 or pos[0] >= self.grid_w:
            return True
        elif pos[1] < 0 or pos[1] >= self.grid_h:
            return True
        # 判断贪吃蛇头是否碰到了身体
        elif list(pos) in [list(item) for item in self.prev_pos[:-1]]:
            return True
        else:
            return False

    def reset(self):
        self.pos = np.array([self.grid_w // 2, self.grid_h // 2]).astype('int')
        self.dir = random.choice(self.dirs)
        self.prev_pos = [np.array([self.grid_w // 2, self.grid_h // 2]).astype('int')] * (self.init_len + 1)
        self.len = self.init_len

    def eat_apple(self):
        # self.len += 1
        pass

    def __len__(self):
        return self.len + 1


class Apple(object):  # 定义苹果出现的地方
    def __init__(self, grid_w, grid_h):
        self.pos = None
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.score = 0
        self.reset()

    def eaten(self):
        self.score += 1
        self.gen_apple_pos()

    def reset(self):
        self.score = 0
        self.gen_apple_pos()

    def gen_apple_pos(self):
        self.pos = np.random.randint(0, [self.grid_w, self.grid_h], 2)


class SnakeWorldEnv(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    action2dir = {
        0: np.array([-1, 0]),  # left
        1: np.array([1, 0]),  # right
        2: np.array([0, -1]),  # up
        3: np.array([0, 1])  # down
    }

    def __init__(self, grid_w, grid_h, rwd_nothing=0, rwd_dead=-10, rwd_apple=10, max_step=1000):
        self.snake = Snake(grid_w, grid_h)
        self.apple = Apple(grid_w, grid_h)
        self.game_over = False
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.reward_nothing = rwd_nothing
        self.reward_dead = rwd_dead
        self.reward_apple = rwd_apple

        self.state = None
        self.step_num = 0
        self.max_step = max_step
        self.reset_num = -1

        self.action_space = None
        self.observation_space = None
        self._set_space()

        self.reward_info = "dead{}_apple{}".format(rwd_dead, rwd_apple)

    def reset(self):
        self.snake.reset()
        self.apple.reset()
        self.game_over = False
        # self.state = self._xy_to_state(self.snake.pos)
        self.step_num = 0
        self.reset_num += 1
        self.state = self._get_state()

        return self.state

    def step(self, action: np.ndarray):
        self.step_num += 1
        info = {
            "step_num": self.step_num,
            "score": self.apple.score
        }
        # assert self.action_space.contains(action), "invalid action: {}".format(action)

        # numpy 转 int
        action = int(action)

        # 如果在动作空间内,则改变snake的dir,否则dir不变
        if self.action_space.contains(action):
            direction = self.action2dir[action]
            self.snake.move(direction)
        else:
            self.snake.move()

        rwd, done = self._get_reward_and_done(action)

        # 设置状态空间: 蛇头位置,苹果位置,身体(固定)4个位置
        self.state = self._get_state()
        if done:
            self.game_over = True
        return self.state, rwd, done, info

    def close(self):
        pygame.quit()
        exit()

    def _get_state(self):
        """设置状态空间: 蛇头位置,墙右下角的位置(暂弃),苹果位置,身体(固定)4个位置, """
        snake_pos = self.snake.pos
        apple_pos = self.apple.pos
        snake_prev_pos = self.snake.prev_pos[:-1]
        prev_pos_concat = np.concatenate(snake_prev_pos)
        snake_dir = self.snake.dir
        # range_wall = np.array([self.grid_w - 1, self.grid_h - 1])

        # 由于discrete固定0~n, 没有负号,只能改成到上下左右的距离,代替snake.pos .
        # wall_rlt_pos = np.array([0-snake_pos[0], self.grid_w-1-snake_pos[0], 0-snake_pos[1], self.grid_h-1-snake_pos[1]])
        snake_pos_rlt_wall = np.array(
            [snake_pos[0], self.grid_w - 1 - snake_pos[0], snake_pos[1], self.grid_h - 1 - snake_pos[1]])
        state_np = np.concatenate(
            [snake_pos_rlt_wall.flatten(), snake_dir.flatten(), apple_pos.flatten(), prev_pos_concat.flatten()])
        return state_np

    def _get_state_img(self):
        pass

    def _get_reward_and_done(self, action):
        is_dead = self.snake.check_dead(self.snake.pos)
        done = (self.step_num > self.max_step) or is_dead
        reward = self._get_reward(is_dead, action)
        return reward, done

    def _get_reward(self, is_dead, action):
        r = 0
        if is_dead:
            r = self.reward_dead
        elif (self.snake.pos == self.apple.pos).all():
            self.apple.eaten()
            self.snake.eat_apple()
            r = self.reward_apple
        else:
            # dis reward
            r_dis = self._get_reward_dis()

            # dir reward, 如果是无效操作
            direction = self.action2dir[action]
            if ((direction + self.snake.dir) == np.array([0, 0])).all():
                r_dir_rate = -0.5
            # 步数惩罚
            r = r_dis - 1


        return r

    def _get_reward_dis(self):
        snake_pos = self.snake.pos
        apple_pos = self.apple.pos
        dis = (math.sqrt(pow((snake_pos[0] - apple_pos[0]), 2) + pow((apple_pos[1] - snake_pos[1]), 2)))
        reward = (1 / dis) * 0.5
        return reward

    def _get_distance(self, m, n):
        assert len(m) == 2 and len(n) == 2
        if isinstance(m, np.ndarray) and isinstance(n, np.ndarray):
            m, n = m.tolist(), n.tolist()
        dis_x = m[0] - n[0]
        dis_y = m[1] - n[1]
        return math.sqrt(math.pow(dis_x, 2) + math.pow(dis_y, 2))

    def _set_space(self):
        self.action_space = spaces.Discrete(4)
        w, h = self.grid_w, self.grid_h
        # snake_pos, snake_dir, apple_pos, snake_prev_pos, wall_rlt_pos
        obs_dim = [w, w, h, h] + [2, 2] + [w, h] + [w, h] * self.snake.init_len
        self.observation_space = spaces.MultiDiscrete(obs_dim)

        # Dict space
        # obs_space = spaces.Dict(
        #     {
        #         "snake_rlt_wall_pos": spaces.MultiDiscrete([w,w,h,h]),
        #         "snake_dir": spaces.MultiBinary(2),
        #         "apple_pos": spaces.MultiDiscrete([w, h]),
        #         "snake_prev_pos": spaces.MultiDiscrete([w, h]*self.snake.init_len)
        #     }
        # )
        # self.observation_space = obs_space

    def render(self, mode='human', fps=10, display=True):
        if not display:
            os.environ['SDL_VIDEODRIVER'] = "dummy"


class UIBoard(object):
    user_input = {
        pygame.K_LEFT: 0,  # left
        pygame.K_RIGHT: 1,  # right
        pygame.K_UP: 2,  # up
        pygame.K_DOWN: 3,  # down
        pygame.K_SPACE: 4,
        pygame.K_ESCAPE: 5
    }

    def __init__(self, grid_w, grid_h, fps=10):
        self.world = SnakeWorldEnv(grid_w, grid_h)
        self.block_size = 20
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.win_w = 0
        self.win_h = 0
        self.win = None
        self.clock = None
        self.font = None
        self.fps = fps
        self.init_win()

    def init_win(self):

        self.win_w = (self.grid_w + 7) * self.block_size
        self.win_h = self.grid_h * self.block_size

        # os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 0)
        pygame.init()
        self.win = pygame.display.set_mode((self.win_w, self.win_h))

        pygame.display.set_caption("snake")
        self.font = pygame.font.SysFont('arial', 18)
        self.clock = pygame.time.Clock()

    def draw_board(self):
        snake = self.world.snake
        apple = self.world.apple
        block_size = self.block_size
        self.win.fill((0, 0, 0))
        for pos in snake.prev_pos:
            pygame.draw.rect(self.win, (0, 255, 0), (pos[0] * block_size, pos[1] * block_size, block_size, block_size))
        pygame.draw.rect(self.win, (255, 0, 0),
                         (apple.pos[0] * block_size, apple.pos[1] * block_size, block_size, block_size))
        # boundary
        for i in range(self.grid_h):
            pygame.draw.rect(self.win, (255, 255, 255),
                             (self.grid_w * block_size, i * block_size, block_size, block_size))
        # game info
        score_text = self.font.render("Score: {}".format(apple.score), False, (255, 255, 255))
        step_text = self.font.render("Step: {}".format(self.world.step_num), False, (255, 255, 255))
        dead_text = self.font.render("Dead: {}".format(self.world.reset_num), False, (255, 255, 255))

        self.win.blit(score_text, ((self.grid_w + 1) * block_size, 1 * block_size))
        self.win.blit(step_text, ((self.grid_w + 1) * block_size, 3 * block_size))
        self.win.blit(dead_text, ((self.grid_w + 1) * block_size, 5 * block_size))

    def begin_game(self):
        self.draw_board()


if __name__ == '__main__':
    pass
