from algorithms.DQN.DQN import DQN
from algorithms.ppo.ppo import PPO
from game_envs.Snakey import *

class SnakeyMainRecoder(object):
    def __init__(self):
        pass

    @staticmethod
    def play_game_manual():
        board = UIBoard(30, 30, fps=5)
        board.begin_game()
        game_exit = False
        user_input = board.user_input
        clock = board.clock
        fps = board.fps
        paused = False
        while not game_exit:
            action = None
            clock.tick(fps)
            board.draw_board()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key in user_input.keys():
                    user_input_code = user_input[event.key]
                    if user_input_code == 4:  # pause
                        paused = True
                    elif user_input_code == 5:
                        game_exit = True
                        break
                    else:
                        action = user_input_code
            while paused:
                clock.tick(10)
                pygame.event.pump()
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        paused = False

            board.world.step(action)

            game_over = board.world.game_over
            pygame.display.update()

            if game_over:
                board.world.reset()
        pygame.quit()

    @staticmethod
    def play_game_auto(agent):
        board = UIBoard(20, 20, fps=20)
        board.begin_game()
        game_exit = False
        clock = board.clock
        fps = board.fps
        paused = False
        state = board.world.reset()
        while not game_exit:
            game_over = board.world.game_over
            while not game_over:


                clock.tick(fps)
                board.draw_board()
                action = agent.get_action(state, 0)
                state, _, game_over, info = board.world.step(action)
                pygame.display.update()
                # tmp = pygame.surfarray.array3d(pygame.display.get_surface())
                # print(tmp.shape)

                while paused:
                    clock.tick(fps)
                    pygame.event.pump()
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            paused = False
                            game_exit = True
                            game_over = True
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                            paused = False

                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        game_exit = True
                        game_over = True
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and not paused:
                        paused = True
                    if event.type == pygame.KEYDOWN and event.key == pygame.QUIT:
                        board.world.close()
            if game_over:
                state = board.world.reset()
                paused = True

        pygame.quit()

    @staticmethod
    def train_dqn():
        env_snake = SnakeWorldEnv(20, 20, rwd_dead=-50, rwd_apple=50)
        dqn = DQN(env_snake)
        dqn.learn(total_episode=10000)

    @staticmethod
    def inference_dqn():
        check_point = "./data/models/snakey_dqn/episode5000_lr0.001_epsilon0.2_batchsize32_gamma0.9_dead-50_apple50.pth"
        env_snake = SnakeWorldEnv(20, 20)
        dqn = DQN(env_snake)
        dqn.load(check_point)
        dqn.net.eval()
        SnakeyMainRecoder.play_game_auto(dqn)

    @staticmethod
    def train_ppo():
        env_snake = SnakeWorldEnv(20, 20, rwd_dead=-100, rwd_apple=200)
        ppo = PPO(env_snake, epochs=100)
        ppo.learn(model_save_root="./data/models/snakey_ppo")

    @staticmethod
    def inference_ppo():
        check_point = "./data/models/snakey_ppo/epoch99_dead-100_apple200.pt"
        env_snake = SnakeWorldEnv(20, 20, rwd_dead=-100, rwd_apple=200)
        ppo = PPO(env_snake)
        ppo.load_model(check_point)
        ppo.ac.eval()
        SnakeyMainRecoder.play_game_auto(ppo)

if __name__ == '__main__':
    snake_main_recorder = SnakeyMainRecoder()
    snake_main_recorder.inference_ppo()
