from gymnasium.envs.registration import register

register(
    id="snake/SnakeGrid",
    entry_point="snake.envs:SnakeGrid",
)

register(
    id="snake/SnakeGridDiscrete",
    entry_point="snake.envs:SnakeGridDiscrete",
)

register(
    id="snake/SnakeDirectional",
    entry_point="snake.envs:SnakeDirectional",
)

register(
    id="snake/SnakePositional",
    entry_point="snake.envs:SnakePositional",
)
