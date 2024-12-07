import numpy as np
from gymnasium import spaces

from .snake_base import SnakeBase


class SnakePositions(SnakeBase):
    def __init__(self, render_mode=None, size=5):
        super().__init__(render_mode, size)

        # Agent observes the 8 tiles directly surrounding its head, returning a
        # binary result for each tile indicating whether there is danger or not.
        # Additionally, it observes if food is above, below, left, and/or right.
        max_distance_1d = size - 1
        self.observation_space = spaces.Dict(
            {
                # Distance from the head in each direction
                "nearest_danger": spaces.Box(
                    low=np.array([-max_distance_1d] * 4),
                    high=np.array([max_distance_1d] * 4),
                ),
                # The position of the food relative to the head
                "food": spaces.Box(
                    low=np.array([-max_distance_1d] * 2),
                    high=np.array([max_distance_1d] * 2),
                ),
            }
        )

    def get_obs(self):  # type: ignore
        nearest_danger = np.zeros(4)
        food = np.zeros(2)

        if self.dead:
            return {
                "nearest_danger": nearest_danger.astype(np.float32),
                "food": food.astype(np.float32),
            }

        head_coord = self._index_to_coordinate(self._snake_indices[0])
        differences = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])
        for i, relative_position in enumerate(differences):
            distance = 1
            while True:
                diff = distance * relative_position
                if self.dead_index(self._coordinate_to_index(head_coord + diff)):
                    nearest_danger[i] = distance
                    break
                distance += 1

        food_coord = self._index_to_coordinate(self._food_index)
        food = food_coord - head_coord

        return {
            "nearest_danger": nearest_danger.astype(np.float32),
            "food": food.astype(np.float32),
        }

    def get_info(self):
        info = super().get_info()
        board = self._get_board_state().reshape([self.full_size, self.full_size])
        info["board"] = board
        return info
