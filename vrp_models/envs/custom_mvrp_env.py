import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MVRPEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(self):
        super().__init__()
        self.num_nodes = 10
        self.num_vehicles = 3
        
        self.action_space = spaces.Discrete(self.num_vehicles)
        
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(self.num_nodes + self.num_vehicles,), dtype=np.float32
        )
        
        self._state = np.zeros(self.num_nodes + self.num_vehicles, dtype=np.float32)

    def step(self, action):
        reward = -1.0 
        terminated = False
        truncated = False
        info = {}

        new_state_array = self._calculate_new_state(action)
        self.unwrapped.s = new_state_array 
        self._state = new_state_array

        return self._state, reward, terminated, truncated, info

    def _calculate_new_state(self, action):
        return np.random.rand(self.num_nodes + self.num_vehicles).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._state = np.zeros(self.num_nodes + self.num_vehicles, dtype=np.float32)
        self.unwrapped.s = self._state
        return self._state, {}
