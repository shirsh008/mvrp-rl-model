import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

FIXED_DISTANCES = np.array([
    [0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354, 468, 776, 662],
    [548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674, 1016, 868, 1210],
    [776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164, 1130, 788, 1552, 754],
    [696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822, 1164, 560, 1358],
    [582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708, 1050, 674, 1244],
    [274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628, 514, 1050, 708],
    [502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856, 514, 1278, 480],
    [194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320, 662, 742, 856],
    [308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662, 320, 1084, 514],
    [194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388, 274, 810, 468],
    [536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764, 730, 388, 1152, 354],
    [502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114, 308, 650, 274, 844],
    [388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194, 536, 388, 730],
    [354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0, 342, 422, 536],
    [468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536, 342, 0, 764, 194],
    [776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274, 388, 422, 764, 0, 798],
    [662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730, 536, 194, 798, 0]
], dtype=np.float32)

FIXED_DEMANDS = np.array([0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8], dtype=np.float32)

class CustomFixedMVRPEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(self, num_nodes=17, num_vehicles=4, vehicle_capacity=15.0, render_mode=None):
        super(CustomFixedMVRPEnv, self).__init__()
        self.num_nodes = num_nodes
        self.num_vehicles = num_vehicles
        self.max_capacity = vehicle_capacity
        self.render_mode = render_mode
        self.demands = None
        self.vehicle_locations = None
        self.vehicle_capacities = None
        self.visited_mask = None
        self.current_vehicle = 0
        self.action_space = spaces.Discrete(self.num_nodes)
        self.observation_space = spaces.Dict({
            "locations": spaces.Box(low=0.0, high=1.0, shape=(self.num_nodes, 2), dtype=np.float32),
            "demands": spaces.Box(low=0.0, high=15.0, shape=(self.num_nodes,), dtype=np.float32),
            "capacities": spaces.Box(low=0.0, high=self.max_capacity, shape=(self.num_vehicles,), dtype=np.float32),
            "active_locations": spaces.MultiDiscrete([self.num_nodes] * self.num_vehicles),
            "visited": spaces.MultiBinary(self.num_nodes)
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.node_locations = np.zeros((self.num_nodes, 2), dtype=np.float32)
        self.demands = FIXED_DEMANDS.copy()
        self.vehicle_locations = np.zeros(self.num_vehicles, dtype=np.int32)
        self.vehicle_capacities = np.full(self.num_vehicles, self.max_capacity, dtype=np.float32)
        self.visited_mask = np.zeros(self.num_nodes, dtype=np.int8)
        self.visited_mask[0] = 1 
        self.current_vehicle = 0
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        return {
            "locations": self.node_locations,
            "demands": self.demands,
            "capacities": self.vehicle_capacities,
            "active_locations": self.vehicle_locations,
            "visited": self.visited_mask
        }

    def _get_info(self):
        return {
            "action_mask": np.logical_not(self.visited_mask).astype(np.int8),
            "current_vehicle": self.current_vehicle
        }

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        
        reward = 0.0
        terminated = False
        truncated = False
        curr_loc = self.vehicle_locations[self.current_vehicle]
        dist = FIXED_DISTANCES[curr_loc][action]

        if self.visited_mask[action] == 1 and action != 0:
            reward -= 5000.0 
            return self._get_obs(), reward, terminated, truncated, self._get_info()
            
        reward -= float(dist)
        self.vehicle_locations[self.current_vehicle] = action
        
        if action != 0:
            demand = self.demands[action]
            if self.vehicle_capacities[self.current_vehicle] >= demand:
                self.vehicle_capacities[self.current_vehicle] -= demand
                self.visited_mask[action] = 1
                self.demands[action] = 0.0
                reward += 1000.0 
            else:
                reward -= 5000.0 
        else:
            self.vehicle_capacities[self.current_vehicle] = self.max_capacity
            
        if np.all(self.visited_mask == 1):
            terminated = True
            
        self.current_vehicle = (self.current_vehicle + 1) % self.num_vehicles
        return self._get_obs(), reward, terminated, truncated, self._get_info()
