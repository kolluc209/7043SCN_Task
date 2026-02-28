"""
Train a PPO agent with reward shaping on the Chef's Hat card game.
Shaped reward: bonus for reducing hand size (playing cards).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from chefs_hat_env import ChefsHatEnv


class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        shaped_reward = reward

        # Small reward for valid action
        if reward == 0:
            shaped_reward += 0.1

        # Larger signal at game end
        if done:
            if reward > 0:
                shaped_reward += 10
            else:
                shaped_reward -= 10

        return obs, shaped_reward, terminated, truncated, info


def make_env():
    env = ChefsHatEnv(max_matches=3)
    env = RewardShapingWrapper(env)
    return env


env = make_vec_env(make_env, n_envs=1)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
)

print("Training shaped reward agent...")
model.learn(total_timesteps=20000)
model.save("ppo_chefshat_shaped")

print("Training complete. Model saved to 'ppo_chefshat_shaped.zip'")
