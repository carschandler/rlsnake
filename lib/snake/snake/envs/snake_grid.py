import numpy as np
from gymnasium import spaces

from .snake_base import SnakeBase


class SnakeGrid(SnakeBase):
    def __init__(self, render_mode=None, size=5, dtype=np.int32):
        super().__init__(render_mode, size, dtype)

        self.observation_space = spaces.Box(
            low=0,
            high=4,
            shape=[self.full_size, self.full_size],
            dtype=self.dtype,  # type: ignore
        )

    def get_obs(self):
        return self._get_board_state().reshape([self.full_size, self.full_size])
