import sys
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.custom_mvrp_env import CustomFixedMVRPEnv, FIXED_DISTANCES

def train_agent():
    env = CustomFixedMVRPEnv(num_nodes=17, num_vehicles=4, vehicle_capacity=15.0, render_mode=None)
    
    check_env(env, warn=True)
    
    model = PPO("MultiInputPolicy", env, verbose=0, ent_coef=0.05, learning_rate=0.0003)
    print("Starting Model Training...")
    model.learn(total_timesteps=500000, progress_bar=True)
    
    obs, info = env.reset()
    done = False
    
    num_vehicles = env.num_vehicles
    routes = [[0] for _ in range(num_vehicles)]
    route_load_history = [[0.0] for _ in range(num_vehicles)]
    route_distances = [0.0 for _ in range(num_vehicles)]
    current_loads = [0.0 for _ in range(num_vehicles)]
    
    step_count = 0
    max_steps = 100 
    
    print("\n--- Testing Trained Agent ---")
    while not done and step_count < max_steps:
        v = env.current_vehicle
        curr_node = env.vehicle_locations[v]
        demands_before_step = env.demands.copy()

        action, _states = model.predict(obs, deterministic=True)
        next_node = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        if next_node != curr_node:
            dist = FIXED_DISTANCES[curr_node][next_node]
            route_distances[v] += dist

            load_picked_up = 0.0
            if next_node != 0:
                load_picked_up = demands_before_step[next_node]

            current_loads[v] += load_picked_up
            routes[v].append(next_node)
            route_load_history[v].append(current_loads[v])

    for v in range(num_vehicles):
        if routes[v][-1] != 0:
            last_node = routes[v][-1]
            dist = FIXED_DISTANCES[last_node][0]
            route_distances[v] += dist
            routes[v].append(0)
            route_load_history[v].append(current_loads[v])

    total_distance = sum(route_distances)
    total_load = sum(current_loads)

    if step_count >= max_steps:
        print("\n[WARNING]: The agent failed to fulfill all demands and hit the step limit.")

    print(f"Objective: {total_distance:.0f}")
    for v in range(num_vehicles):
        print(f"Route for vehicle {v}:")
        route_str_parts = []
        for i in range(len(routes[v])):
            node = routes[v][i]
            load = route_load_history[v][i]
            route_str_parts.append(f"{node} Load({int(load)})")
            
        route_str = " -> ".join(route_str_parts)
        print(f" {route_str}")
        print(f"Distance of the route: {route_distances[v]:.0f}m")
        print(f"Load of the route: {int(current_loads[v])}\n")

    print(f"Total distance of all routes: {total_distance:.0f}m")
    print(f"Total load of all routes: {int(total_load)}")

if __name__ == "__main__":
    train_agent()
