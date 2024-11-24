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
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class States(Enum):
    EMPTY = 0
    WALL = 1
    HEAD = 2
    TAIL = 3
    FOOD = 4


SYMBOLS = {
    States.EMPTY: " ",
    States.HEAD: {
        Directions.RIGHT: ">",
        Directions.LEFT: "<",
        Directions.UP: "^",
        Directions.DOWN: "v",
    },
    States.TAIL: "*",
    States.FOOD: "$",
    States.WALL: "#",
}
DEAD_SYMBOL = "x"

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
        self._dead = False

        # Each square in the grid can be:
        num_discrete_values_per_location = np.full_like(all_indices, 5)

        # self.observation_space = spaces.MultiDiscrete(
        #     nvec=num_discrete_values_per_location,
        #     dtype=np.uint8,
        # )
        self._dtype = np.float32
        self.observation_space = spaces.Box(
            low=0,
            high=4,
            shape=[self.full_size, self.full_size],
            dtype=self._dtype,
        )
        # spaces.Tuple([spaces.Discrete(n=len(States)) for _ in range(size**2)])

        # We have 3 actions, corresponding to turning left, right, or keeping straight
        self.action_space = spaces.Discrete(len(Actions))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._initialize_blank_board()

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_board_state(self):
        board = self._blank_board.copy()
        board[self._food_index] = States.FOOD.value
        board[self._snake_indices] = States.TAIL.value
        board[self._snake_indices[0]] = States.HEAD.value

        self._board = board

        return self._board

    def _get_obs(self):
        return self._get_board_state().reshape([self.full_size, self.full_size])

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._index_to_coordinate(self._food_index)
                - self._index_to_coordinate(self._snake_indices[0]),
                ord=1,
            ),
            "current_direction": np.int64(self._current_direction_value),
            "snake_length": np.int64(self._snake_length),
        }

    # def _action_to_direction(self, action_value: int) -> int:
    #     return (self._current_direction_value + action_value) % len(Directions)

    def _action_to_direction(self, action_value: int) -> int:
        # If snake tries to move against its current direction, it just
        # continues in that direction.
        # TODO: see what happens if we remove this
        if (int(action_value) + 2) % 4 == self._current_direction_value:
            return self._current_direction_value
        return action_value

    def _direction_to_delta(self, direction_value: int):
        return {
            Directions.RIGHT.value: np.array([0, 1]),
            Directions.UP.value: np.array([-1, 0]),
            Directions.LEFT.value: np.array([0, -1]),
            Directions.DOWN.value: np.array([1, 0]),
        }[direction_value]

    def _initialize_blank_board(self):
        board = np.full(self.full_size**2, States.EMPTY.value, dtype=self._dtype)
        board[self._wall_indices] = States.WALL.value
        self._blank_board = board
        self._board = board.copy()

    def _coordinate_to_index(self, coordinate):
        return np.ravel_multi_index(
            coordinate.tolist(), dims=[self.full_size, self.full_size]
        )

    def _index_to_coordinate(self, index: Index) -> Coordinate:
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
        self._snake_array[0] = self._coordinate_to_index(initial_coordinate)
        self._snake_length = 1
        self._update_snake_indices()
        # Start the snake in a random direction
        self._current_direction_value = self.np_random.choice(
            np.array(Directions)
        ).value

        self._spawn_new_food()

        observation = self._get_obs()
        info = self._get_info()

        self.render(observation, self._current_direction_value, dead=False)

        return observation, info

    def step(self, action: int):
        # Map the action to the direction we walk in
        # RIGHT, UP, LEFT, DOWN
        direction_value = self._action_to_direction(action)
        self._current_direction_value: int = direction_value
        delta = self._direction_to_delta(direction_value)

        # TODO: can optimize this by directly calculating the index
        next_head_coordinate = self._index_to_coordinate(self._snake_indices[0]) + delta
        next_head_index = self._coordinate_to_index(next_head_coordinate)

        # TODO: if we allow the agent to run into walls and receive negative
        # rewards as a result, then the walls need to be in the state space...
        # otherwise, we need to prevent them from being accessible by either
        # clipping resulting states or re-drawing actions (not sure if this is
        # possible). If we just clip states, then it will be possible to get
        # stuck, and we could introduce a terminal state if stuck for more than
        # x steps.
        self._dead = self._dead_coordinate(next_head_index)
        self._won = self._snake_length == self.playable_size**2

        got_food = next_head_index == self._food_index
        self._step_snake(next_head_index, got_food)
        if got_food:
            self._spawn_new_food()

        # An episode is done if the snake has hit a wall, itself, or has won
        terminated = self._dead or self._won
        reward = 1 if got_food else -1 if self._dead else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        self.render(observation, self._current_direction_value, self._dead)

        truncated = False

        return observation, reward, terminated, truncated, info

    def render(self, observation=None, current_direction=None, dead=None):
        if self.render_mode == "ansi":
            if all([x is not None for x in [observation, current_direction, dead]]):
                return self._render_as_string(observation, current_direction, dead)
            else:
                return self._render_as_string(
                    self._get_obs(), self._current_direction_value, self._dead
                )

    @staticmethod
    def _update_string_board_from_observation(
        observation_array, string_array, state_value, current_direction, dead
    ):
        if state_value == States.HEAD.value:
            string_array[observation_array == state_value] = SYMBOLS[States.HEAD][
                Directions(current_direction)
            ]
        else:
            string_array[observation_array == state_value] = SYMBOLS[
                States(state_value)
            ]

    @staticmethod
    def _render_as_string(observation, current_direction, dead):
        string_board = np.full_like(
            observation, SYMBOLS[States.EMPTY], dtype=np.dtype("<U1")
        )

        for state_value in np.unique(observation):
            SnakeEnv._update_string_board_from_observation(
                observation, string_board, state_value, current_direction, dead
            )

        if dead:
            string_board[
                np.isin(observation, [States.HEAD.value, States.TAIL.value])
            ] = DEAD_SYMBOL

        return "\n".join(" ".join(char for char in row) for row in string_board)

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
