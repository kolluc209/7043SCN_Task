"""
Train Sparse/Delayed Reward Variant Agents

This script trains PPO agents on the Chef's Hat card game with various
reward shaping and auxiliary reward strategies. The focus is on handling
sparse, delayed rewards through:

1. Multiple reward shaping techniques
2. Auxiliary rewards and credit assignment
3. Curriculum learning
4. Learning from trajectories

Author: Students (Sparse/Delayed Reward Variant - ID mod 7 = 2)
"""

import sys
import os
import json
import argparse
from datetime import datetime
import numpy as np
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

# Import custom modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from chefs_hat_env import ChefsHatEnv
from reward_shaping_strategies import create_reward_shaper
from auxiliary_rewards import create_auxiliary_wrapper


class TrainingMetricsCallback(BaseCallback):
    """Custom callback to track training metrics."""

    def __init__(self, eval_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_interval = eval_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        # Track episodic rewards and lengths
        dones = self.model.get_attr("dones") if hasattr(self.model, "get_attr") else [False]

        for i in range(self.model.env.num_envs):
            self.current_episode_reward += self.locals.get("rewards", [0])[i]
            self.current_episode_length += 1

            # Check if episode is done
            if i < len(dones) and dones[i]:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.current_episode_reward = 0
                self.current_episode_length = 0

        return True

    def get_metrics(self):
        """Return training metrics."""
        if len(self.episode_rewards) == 0:
            return {}

        return {
            "mean_episode_reward": np.mean(self.episode_rewards[-100:]),
            "max_episode_reward": np.max(self.episode_rewards),
            "min_episode_reward": np.min(self.episode_rewards),
            "std_episode_reward": np.std(self.episode_rewards[-100:]),
            "mean_episode_length": np.mean(self.episode_lengths[-100:]),
            "total_episodes": len(self.episode_rewards),
        }


def create_env(
    max_matches: int = 3,
    reward_shaping: str = "hybrid",
    auxiliary_rewards: str = "multitask",
    **kwargs
) -> gym.Env:
    """
    Create a Chef's Hat environment with reward shaping and auxiliary rewards.

    Parameters
    ----------
    max_matches : int
        Number of matches per game
    reward_shaping : str
        Reward shaping strategy
    auxiliary_rewards : str
        Auxiliary reward strategy
    **kwargs : dict
        Additional arguments

    Returns
    -------
    gym.Env
        Wrapped environment
    """
    env = ChefsHatEnv(max_matches=max_matches)

    # Apply auxiliary rewards first
    if auxiliary_rewards:
        env = create_auxiliary_wrapper(auxiliary_rewards, env, **kwargs)

    # Then apply reward shaping
    if reward_shaping:
        env = create_reward_shaper(reward_shaping, env, **kwargs)

    return env


def train_agent(
    output_dir: str,
    reward_shaping: str = "hybrid",
    auxiliary_rewards: str = "multitask",
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    verbose: int = 1,
):
    """
    Train a PPO agent with specified reward shaping and auxiliary strategies.

    Parameters
    ----------
    output_dir : str
        Directory to save outputs
    reward_shaping : str
        Reward shaping strategy ('none', 'hand_size', 'progress', 'hybrid', 'curriculum')
    auxiliary_rewards : str
        Auxiliary reward strategy ('none', 'curiosity', 'entropy', 'trajectory', 'multitask')
    total_timesteps : int
        Total training timesteps
    learning_rate : float
        Learning rate for PPO
    n_steps : int
        Rollout length
    batch_size : int
        Batch size
    n_epochs : int
        Number of training epochs per update
    gamma : float
        Discount factor
    gae_lambda : float
        GAE lambda parameter
    clip_range : float
        PPO clip range
    ent_coef : float
        Entropy coefficient
    vf_coef : float
        Value function coefficient
    verbose : int
        Verbosity level

    Returns
    -------
    dict
        Training results and metadata
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create environment
    def make_env_fn():
        return create_env(
            max_matches=3,
            reward_shaping=reward_shaping,
            auxiliary_rewards=auxiliary_rewards,
        )

    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Reward Shaping: {reward_shaping}")
    print(f"Auxiliary Rewards: {auxiliary_rewards}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Gamma: {gamma}, GAE Lambda: {gae_lambda}")
    print(f"Output Dir: {output_dir}")
    print(f"{'='*60}\n")

    # Create vectorized environment
    env = make_vec_env(make_env_fn, n_envs=1)

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=verbose,
        tensorboard_log=os.path.join(output_dir, "tensorboard"),
    )

    # Create metric callback
    metrics_callback = TrainingMetricsCallback(eval_interval=1000)

    # Train
    print("Starting training...\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=metrics_callback,
        progress_bar=True,
    )

    print("\nTraining complete!")

    # Save model
    model_path = os.path.join(output_dir, "ppo_model")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # Save training configuration and metrics
    metrics = metrics_callback.get_metrics()
    config = {
        "reward_shaping": reward_shaping,
        "auxiliary_rewards": auxiliary_rewards,
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "timestamp": datetime.now().isoformat(),
    }

    metadata = {**config, **metrics}

    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

    env.close()

    return metadata


def train_comparison_agents(output_base_dir: str = "experiments"):
    """
    Train agents with different reward shaping and auxiliary reward combinations.

    Parameters
    ----------
    output_base_dir : str
        Base directory for all experiments
    """

    os.makedirs(output_base_dir, exist_ok=True)

    # Define experiment configurations
    experiments = [
        # Baseline: no shaping, no auxiliary
        {
            "name": "baseline_no_shaping",
            "reward_shaping": "none",
            "auxiliary_rewards": "none",
        },
        # Baseline with only auxiliary
        {
            "name": "baseline_multitask",
            "reward_shaping": "none",
            "auxiliary_rewards": "multitask",
        },
        # Hand size shaping
        {
            "name": "hand_size_shaping",
            "reward_shaping": "hand_size",
            "auxiliary_rewards": "none",
        },
        # Hybrid shaping with multitask
        {
            "name": "hybrid_multitask",
            "reward_shaping": "hybrid",
            "auxiliary_rewards": "multitask",
        },
        # Hybrid shaping with trajectory
        {
            "name": "hybrid_trajectory",
            "reward_shaping": "hybrid",
            "auxiliary_rewards": "trajectory",
        },
        # Curriculum learning
        {
            "name": "curriculum_multitask",
            "reward_shaping": "curriculum",
            "auxiliary_rewards": "multitask",
        },
        # Curiosity-driven
        {
            "name": "hand_size_curiosity",
            "reward_shaping": "hand_size",
            "auxiliary_rewards": "curiosity",
        },
    ]

    results = []

    for exp_config in experiments:
        exp_name = exp_config.pop("name")
        output_dir = os.path.join(output_base_dir, exp_name)

        print(f"\n{'#'*70}")
        print(f"# Experiment: {exp_name}")
        print(f"{'#'*70}")

        try:
            metadata = train_agent(output_dir, total_timesteps=50000, **exp_config)
            metadata["experiment_name"] = exp_name
            results.append(metadata)
            print(f"\n✓ {exp_name} completed successfully")
        except Exception as e:
            print(f"\n✗ {exp_name} failed with error: {e}")
            import traceback
            traceback.print_exc()

    # Save comparison results
    comparison_path = os.path.join(output_base_dir, "comparison_results.json")
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'='*70}")
    print("Comparison Results Summary")
    print(f"{'='*70}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {len(results)}")
    print(f"Results saved to: {comparison_path}")

    # Print summary table
    if results:
        print(f"\n{'Experiment':<30} {'Avg Reward':<15} {'Max Reward':<15}")
        print("-" * 60)
        for r in results:
            exp_name = r.get("experiment_name", "Unknown")
            avg_reward = r.get("mean_episode_reward", "N/A")
            max_reward = r.get("max_episode_reward", "N/A")
            if isinstance(avg_reward, (int, float)):
                print(
                    f"{exp_name:<30} {avg_reward:>14.2f} {max_reward:>14.2f}"
                )
            else:
                print(f"{exp_name:<30} {avg_reward:>14} {max_reward:>14}")


def main():
    parser = argparse.ArgumentParser(description="Train sparse reward agents")
    parser.add_argument(
        "--mode",
        type=str,
        default="comparison",
        choices=["single", "comparison"],
        help="Training mode",
    )
    parser.add_argument(
        "--reward-shaping",
        type=str,
        default="hybrid",
        choices=["none", "hand_size", "progress", "hybrid", "curriculum"],
        help="Reward shaping strategy",
    )
    parser.add_argument(
        "--auxiliary",
        type=str,
        default="multitask",
        choices=["none", "curiosity", "entropy", "trajectory", "multitask"],
        help="Auxiliary reward strategy",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Output directory",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level",
    )

    args = parser.parse_args()

    if args.mode == "comparison":
        train_comparison_agents(output_base_dir=args.output_dir)
    else:
        output_dir = os.path.join(
            args.output_dir,
            f"{args.reward_shaping}_{args.auxiliary}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        train_agent(
            output_dir,
            reward_shaping=args.reward_shaping,
            auxiliary_rewards=args.auxiliary,
            total_timesteps=args.timesteps,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
