import gymnasium as gym
import numpy as np


class ClipReward(gym.RewardWrapper):
    def __init__(self, env, min_reward: np.float64, max_reward: np.float64):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)

    def reward(self, reward):
        return np.clip(np.float64(reward), self.min_reward, self.max_reward)
