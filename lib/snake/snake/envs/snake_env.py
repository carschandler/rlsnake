from enum import Enum
from typing import TypeAlias

import gymnasium as gym

# import pygame
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray


class Directions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class Actions(Enum):
    TURN_CW = -1
    STRAIGHT = 0
    TURN_CCW = 1 


class States(Enum):
    EMPTY = 0
    HEAD = 1
    TAIL = 2
    FOOD = 3
    WALL = 4


SYMBOLS = {
    # TODO: directional + dead head variations
    States.EMPTY: " ",
    States.HEAD: "@",
    States.TAIL: "*",
    States.FOOD: "$",
    States.WALL: "#",
}

Coordinate: TypeAlias = NDArray[np.int64]
Index: TypeAlias = np.int64
IntArray: TypeAlias = NDArray[np.int64]


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        # The width of the playable square grid
        self.playable_size = size
        # The width of the playable grid plus the surrounding walls
        self.full_size = size + 2

        all_indices: IntArray = np.reshape(
            np.arange(self.full_size**2), [self.full_size, self.full_size]
        )
        self._playable_indices: IntArray = all_indices[1:-1, 1:-1].flatten()
        self._wall_indices: IntArray = np.setdiff1d(all_indices, self._playable_indices)

        # Each square in the grid can be:
        # 0: EMPTY
        # 1: HEAD
        # 2: TAIL
        # 3: FOOD
        self.observation_space = spaces.Dict(
            dict(
                snake_indices=spaces.Sequence(spaces.Discrete(n=self.full_size**2)),
                # TODO: should this be limited to playable_size and we convert between indices?
                food_index=spaces.Discrete(n=self.full_size**2),
            )
        )
        # spaces.Tuple([spaces.Discrete(n=len(States)) for _ in range(size**2)])

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(len(Actions))

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._initialize_blank_board_str_array()

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
            ),
            "current_direction": Directions(self._current_direction_value),
            "snake_length": self._snake_length,
        }

    def _action_to_direction(self, action_value: int) -> int:
        return (self._current_direction_value + action_value) % len(Directions)

    def _direction_to_delta(self, direction_value: int):
        return {
            Directions.RIGHT.value: np.array([0, 1]),
            Directions.UP.value: np.array([-1, 0]),
            Directions.LEFT.value: np.array([0, -1]),
            Directions.DOWN.value: np.array([1, 0]),
        }[direction_value]

    def _initialize_blank_board_str_array(self):
        board = np.full(self.full_size**2, SYMBOLS[States.EMPTY])
        board[self._wall_indices] = SYMBOLS[States.WALL]
        self._blank_board_str_array = board

    def _c2i(self, coordinate: Coordinate) -> Index:
        return np.ravel_multi_index(
            coordinate.tolist(), dims=[self.full_size, self.full_size]
        )

    def _i2c(self, index: Index) -> Coordinate:
        return np.array(
            np.unravel_index(index, [self.full_size, self.full_size]), dtype=np.int64
        )

    def _spawn_new_food(self):
        # Choose the food location uniformly at random such that it isn't on top
        # of the snake
        valid_food_indices = self._playable_indices[
            ~np.isin(self._playable_indices, self._snake_indices)
        ]
        self._food_index: np.int64 = self.np_random.choice(valid_food_indices)

    def _dead_coordinate(self, index: Index):
        return np.any(
            np.isin(index, self._wall_indices) | np.isin(index, self._snake_indices)
        )

    def _step_snake(self, next_head_index: np.int64, food: bool):
        self._snake_array[1:] = self._snake_array[:-1]
        self._snake_array[0] = next_head_index
        if food:
            self._snake_length += 1
        self._update_snake_indices()

    def _update_snake_indices(self):
        self._snake_indices = self._snake_array[: self._snake_length]

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Start the snake in the middle
        initial_coordinate = np.array([self.full_size // 2, self.full_size // 2])
        self._snake_array: IntArray = np.full([self.playable_size**2], -1)
        self._snake_array[0] = self._c2i(initial_coordinate)
        self._snake_length = 1
        self._update_snake_indices()
        # Start the snake in a random direction
        self._current_direction_value = self.np_random.choice(np.array(Directions)).value

        self._spawn_new_food()

        observation = self._get_obs()
        info = self._get_info()

        self.render()

        return observation, info

    def step(self, action_value: int):
        # Map the action to the direction we walk in
        direction_value = self._action_to_direction(action_value)
        self._current_direction_value: int = direction_value
        delta = self._direction_to_delta(direction_value)

        # TODO: can optimize this by directly calculating the index
        next_head_coordinate = self._i2c(self._snake_indices[0]) + delta
        next_head_index = self._c2i(next_head_coordinate)

        # TODO: if we allow the agent to run into walls and receive negative
        # rewards as a result, then the walls need to be in the state space...
        # otherwise, we need to prevent them from being accessible by either
        # clipping resulting states or re-drawing actions (not sure if this is
        # possible). If we just clip states, then it will be possible to get
        # stuck, and we could introduce a terminal state if stuck for more than
        # x steps.
        dead = self._dead_coordinate(next_head_index)
        won = self._snake_length == self.playable_size**2

        got_food = next_head_index == self._food_index
        self._step_snake(next_head_index, got_food)
        if got_food:
            self._spawn_new_food()

        # An episode is done if the snake has hit a wall, itself, or has won
        terminated = dead or won
        reward = 1 if got_food else -1 if dead else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        self.render()

        truncated = False

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self._render_as_string()

    def _render_as_string(self):
        board = self._blank_board_str_array.copy()
        board[self._food_index] = SYMBOLS[States.FOOD]
        board[self._snake_indices] = SYMBOLS[States.TAIL]
        board[self._snake_indices[0]] = SYMBOLS[States.HEAD]

        return "\n".join(
            " ".join(char for char in row)
            for row in board.reshape(self.full_size, self.full_size)
        )

    # def _render_frame(self):
    #     if self.window is None and self.render_mode == "human":
    #         pygame.init()
    #         pygame.display.init()
    #         self.window = pygame.display.set_mode((self.window_size, self.window_size))
    #     if self.clock is None and self.render_mode == "human":
    #         self.clock = pygame.time.Clock()
    #
    #     canvas = pygame.Surface((self.window_size, self.window_size))
    #     canvas.fill((255, 255, 255))
    #     pix_square_size = (
    #         self.window_size / self.size
    #     )  # The size of a single grid square in pixels
    #
    #     # First we draw the target
    #     pygame.draw.rect(
    #         canvas,
    #         (255, 0, 0),
    #         pygame.Rect(
    #             pix_square_size * self._target_location,
    #             (pix_square_size, pix_square_size),
    #         ),
    #     )
    #     # Now we draw the agent
    #     pygame.draw.circle(
    #         canvas,
    #         (0, 0, 255),
    #         (self._agent_location + 0.5) * pix_square_size,
    #         pix_square_size / 3,
    #     )
    #
    #     # Finally, add some gridlines
    #     for x in range(self.size + 1):
    #         pygame.draw.line(
    #             canvas,
    #             0,
    #             (0, pix_square_size * x),
    #             (self.window_size, pix_square_size * x),
    #             width=3,
    #         )
    #         pygame.draw.line(
    #             canvas,
    #             0,
    #             (pix_square_size * x, 0),
    #             (pix_square_size * x, self.window_size),
    #             width=3,
    #         )
    #
    #     if self.render_mode == "human":
    #         # The following line copies our drawings from `canvas` to the visible window
    #         self.window.blit(canvas, canvas.get_rect())
    #         pygame.event.pump()
    #         pygame.display.update()
    #
    #         # We need to ensure that human-rendering occurs at the predefined framerate.
    #         # The following line will automatically add a delay to
    #         # keep the framerate stable.
    #         self.clock.tick(self.metadata["render_fps"])
    #     else:  # rgb_array
    #         return np.transpose(
    #             np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
    #         )

    # def close(self):
    #     if self.window is not None:
    #         pygame.display.quit()
    #         pygame.quit()
