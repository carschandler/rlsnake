from gymnasium.envs.registration import register

register(
    id="snake/Snake-v0",
    entry_point="snake.envs:SnakeEnv",
)
