from enum import Enum
from typing import TypeAlias

import gymnasium as gym

# import pygame
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray


class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class States(Enum):
    EMPTY = 0
    HEAD = 1
    TAIL = 2
    FOOD = 3


Coordinate: TypeAlias = NDArray[np.int64]


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid

        # Each square in the grid can be:
        # 0: EMPTY
        # 1: HEAD
        # 2: TAIL
        # 3: FOOD
        self.observation_space = spaces.Dict(
            dict(
                snake_indices=spaces.Sequence(spaces.Discrete(n=size**2)),
                food_index=spaces.Discrete(n=size**2),
            )
        )
        # spaces.Tuple([spaces.Discrete(n=len(States)) for _ in range(size**2)])

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0]),
            Actions.UP.value: np.array([0, 1]),
            Actions.LEFT.value: np.array([-1, 0]),
            Actions.DOWN.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return dict(
            snake_indices=tuple(self._snake_indices), food_index=self._food_index
        )

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._i2c(self._food_index) - self._i2c(self._snake_indices[0]), ord=1
            )
        }

    def _c2i(self, coordinate: NDArray[np.int64]) -> np.int64:
        return np.ravel_multi_index(coordinate.tolist(), dims=[self.size, self.size])

    def _i2c(self, index: np.int64) -> NDArray[np.int64]:
        return np.array(np.unravel_index(index, [self.size, self.size]), dtype=np.int64)

    def _spawn_new_food(self):
        # Choose the food location uniformly at random such that it isn't on top
        # of the snake
        not_snake = [i for i in range(self.size**2) if i not in self._snake_indices]
        self._food_index: np.int64 = self.np_random.choice(not_snake)

    def _dead_coordinate(self, coordinate: NDArray[np.int64]):
        return (
            np.any((coordinate >= self.size) | (coordinate < 0))
            or self._c2i(coordinate) in self._snake_indices
        )

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Start the snake in the middle
        initial_coordinate = np.array([self.size // 2, self.size // 2])
        self._snake_indices: list[np.int64] = [self._c2i(initial_coordinate)]

        self._spawn_new_food()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # We use `np.clip` to make sure we don't leave the grid
        next_head_coordinate = self._i2c(self._snake_indices[0]) + direction

        # TODO: if we allow the agent to run into walls and receive negative
        # rewards as a result, then the walls need to be in the state space...
        # otherwise, we need to prevent them from being accessible by either
        # clipping resulting states or re-drawing actions (not sure if this is
        # possible). If we just clip states, then it will be possible to get
        # stuck, and we could introduce a terminal state if stuck for more than
        # x steps.
        dead = self._dead_coordinate(next_head_coordinate)

        # An episode is done if the snake has hit a wall, itself, or has won
        terminated = dead or np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
