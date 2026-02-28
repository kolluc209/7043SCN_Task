"""
Evaluate the trained PPO agent on the Chef's Hat card game.
Uses the custom Gymnasium wrapper (ChefsHatEnv).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from stable_baselines3 import PPO
from chefs_hat_env import ChefsHatEnv

# Loading the trained model
model = PPO.load("ppo_chefshat_shaped")

# Creating evaluation environment
env = ChefsHatEnv(max_matches=3)

episodes = 50
wins = 0
total_rewards = []

print(f"Evaluating agent over {episodes} episodes...\n")

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    total_rewards.append(total_reward)
    if total_reward > 0:
        wins += 1

print(f"Results over {episodes} episodes:")
print(f"  Win rate: {wins/episodes:.2f}")
print(f"  Avg reward: {np.mean(total_rewards):.2f}")
print(f"  Min reward: {np.min(total_rewards):.2f}")
print(f"  Max reward: {np.max(total_rewards):.2f}")
