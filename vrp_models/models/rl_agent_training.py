import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envs.custom_mvrp_env import MVRPEnv

def train_model():
    env = MVRPEnv()
    
    check_env(env, warn=True)

    print("Training MVRP RL Agent...")
    model = PPO("MlpPolicy", env, verbose=1)
    
    model.learn(total_timesteps=10000)
    
    model.save("mvrp_ppo_model")
    print("Model saved successfully.")

if __name__ == '__main__':
    train_model()
