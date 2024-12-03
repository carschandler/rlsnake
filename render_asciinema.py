import time

import torch
from snake.envs.snake_env import SnakeEnv

render = SnakeEnv._render_as_string


def render_observations(
    observations: torch.Tensor, directions, done, filepath: str, fps=5
):
    with open(filepath, "w") as f:
        f.write(
            rf'{{"version": 2, "width": {observations.shape[-1]}, "height":'
            rf' {observations.shape[-2]}, "timestamp": {int(time.time())}}}' + "\n"
        )
        t = 0
        for observation, direction, done in zip(observations, directions, done):
            t += 1 / fps
            board_string = render(observation.squeeze(), direction.item(), done.item())
            board_string = board_string.replace("\n", r"\r\n")
            f.write(rf'[{t}, "o", "\u001b[2J{board_string}\r\n"]' + "\n")
