"""
Train a PPO baseline agent on the Chef's Hat card game.
Uses the custom Gymnasium wrapper around the Chef's Hat engine.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Import our custom Gymnasium environment
from chefs_hat_env import ChefsHatEnv


def make_env():
    """Factory function for creating the environment."""
    return ChefsHatEnv(max_matches=3)


# Create vectorized environment
env = make_vec_env(make_env, n_envs=1)

print("Environment created successfully!")
print(f"  Observation space: {env.observation_space}")
print(f"  Action space: {env.action_space}")

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    ent_coef=0.01,
)

print("\nStarting training...")
model.learn(total_timesteps=20000)

model.save("ppo_chefshat_baseline")
print("\nTraining complete! Model saved to 'ppo_chefshat_baseline.zip'")

# Quick evaluation
print("\nRunning quick evaluation (5 games)...")
eval_env = ChefsHatEnv(max_matches=3)
wins = 0
total_rewards = []

for game in range(5):
    obs, info = eval_env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    total_rewards.append(total_reward)
    if total_reward > 0:
        wins += 1
    print(f"  Game {game + 1}: reward = {total_reward:.1f}")

print(f"\nEvaluation results:")
print(f"  Avg reward: {np.mean(total_rewards):.2f}")
print(f"  Wins (positive reward): {wins}/5")
