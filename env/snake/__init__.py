from gymnasium.envs.registration import register

register(
    id="snake/GridWorld-v0",
    entry_point="snake.envs:GridWorldEnv",
)
