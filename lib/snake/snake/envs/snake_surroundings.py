import numpy as np
from gymnasium import spaces

from .snake_base import SnakeBase


class SnakeSurroundings(SnakeBase):
    def __init__(self, render_mode=None, size=5):
        super().__init__(render_mode, size)

        # Agent observes the 8 tiles directly surrounding its head, returning a
        # binary result for each tile indicating whether there is danger or not.
        # Additionally, it observes if food is above, below, left, and/or right.
        self.observation_space = spaces.Dict(
            {
                "danger": spaces.MultiBinary(8),
                "food": spaces.MultiBinary(4),
            }
        )

    def get_obs(self):  # type: ignore
        danger = np.zeros(8)
        food = np.zeros(4)

        head_coord = self._index_to_coordinate(self._snake_indices[0])
        for i, relative_position in enumerate(
            [
                # Start with tile to the right, and wrap CCW
                [0, 1],
                [-1, 1],
                [-1, 0],
                [-1, -1],
                [0, -1],
                [1, -1],
                [1, 0],
                [1, 1],
            ]
        ):
            danger[i] = self.dead_index(
                self._coordinate_to_index(head_coord + relative_position)
            )

        food_coord = self._index_to_coordinate(self._food_index)
        # Right
        if food_coord[1] > head_coord[1]:
            food[0] = 1
        # Above
        if food_coord[0] < head_coord[0]:
            food[1] = 1
        # Left
        if food_coord[1] < head_coord[1]:
            food[2] = 1
        # Below
        if food_coord[0] > head_coord[0]:
            food[3] = 1

        return {
            "danger": danger.astype(np.float32),
            "food": food.astype(np.float32),
        }

    def get_info(self):
        info = super().get_info()
        board = self._get_board_state().reshape([self.full_size, self.full_size])
        info["board"] = board
        return info
