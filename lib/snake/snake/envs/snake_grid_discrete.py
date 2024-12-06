import numpy as np
from gymnasium.spaces import MultiDiscrete

from .snake_base import States
from .snake_grid import SnakeGrid


class SnakeGridDiscrete(SnakeGrid):
    def __init__(self, render_mode=None, size=5, dtype=np.uint8):
        super().__init__(render_mode, size, dtype)

        num_discrete_values_per_location = np.full_like(
            self.get_wrapper_attr("all_indices"), len(States)
        )

        self.observation_space = MultiDiscrete(
            nvec=num_discrete_values_per_location,
            dtype=self.dtype,  # type: ignore
        )

    @staticmethod
    def _render_as_string_onehot(observation, current_direction, dead):
        obs_2d = np.full(observation.shape[-2:], States.EMPTY.value)
        for state_val, state_mask in enumerate(observation):
            obs_2d[np.asarray(state_mask, dtype=bool)] = state_val

        return SnakeGrid._render_as_string(obs_2d, current_direction, dead)
