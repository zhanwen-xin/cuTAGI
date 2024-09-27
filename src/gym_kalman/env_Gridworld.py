import gymnasium as gym
from gymnasium import spaces
import numpy as np
class GridworldEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.observation_space = spaces.Discrete(grid_size * grid_size)
        self.action_space = spaces.Discrete(4)
        self.reward_range = (-1, 1)
        self.reset()

    def reset(self):
        self.state = 0
        self.done = False
        self.info = {}
        return self.state, self.info

    def step(self, action):
        row, col = divmod(self.state, self.grid_size)
        if action == 0:
            'Up'
            row = max(row - 1, 0)
        elif action == 1:
            'Down'
            row = min(row + 1, self.grid_size - 1)
        elif action == 2:
            'Left'
            col = max(col - 1, 0)
        elif action == 3:
            'Right'
            col = min(col + 1, self.grid_size - 1)

        self.state = row * self.grid_size + col

        self.done = self.state == self.grid_size * self.grid_size - 1
        if self.done:
            self.state = None

        reward = -1.0

        return self.state, reward, self.done, False, self.info

    def render(self, mode="human"):
        pass

    def close(self):
        pass
