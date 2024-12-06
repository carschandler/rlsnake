import numpy as np
from gymnasium import Wrapper
from gymnasium.spaces import MultiDiscrete

from ..envs.snake_env import States


class DiscreteObservations(Wrapper):
    def __init__(self, env):
        super().__init__(env)

        num_discrete_values_per_location = np.full_like(
            self.get_wrapper_attr("all_indices"), len(States)
        )

        self.observation_space = MultiDiscrete(
            nvec=num_discrete_values_per_location,
            dtype=self.unwrapped.dtype,
        )

        self._render = self.get_wrapper_attr("render")

    def render(self, observation=None, current_direction=None, dead=None):
        if observation is None:
            observation = self.get_obs()

        obs_2d = np.full(observation.shape[-2:], States.EMPTY.value)
        for state_val, state_mask in enumerate(observation):
            obs_2d[np.asarray(state_mask, dtype=bool)] = state_val

        return self._render(obs_2d, current_direction, dead)
