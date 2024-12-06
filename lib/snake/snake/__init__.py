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
    id="snake/SnakeSurroundings",
    entry_point="snake.envs:SnakeSurroundings",
)

register(
    id="snake/SnakePositions",
    entry_point="snake.envs:SnakePositions",
)
