"""
Evaluation Script for Sparse/Delayed Reward Agents

This script evaluates trained agents with various metrics:
- Win rate
- Average score
- Performance consistency
- Learning dynamics
- Comparison across strategies

Author: Students (Sparse/Delayed Reward Variant - ID mod 7 = 2)
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

from stable_baselines3 import PPO
import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from chefs_hat_env import ChefsHatEnv
from reward_shaping_strategies import create_reward_shaper
from auxiliary_rewards import create_auxiliary_wrapper


class AgentEvaluator:
    """Evaluates trained agents across various metrics."""

    def __init__(
        self,
        model_path: str,
        reward_shaping: str = "none",
        auxiliary_rewards: str = "none",
    ):
        """
        Initialize evaluator.

        Parameters
        ----------
        model_path : str
            Path to saved model (.zip file)
        reward_shaping : str
            Reward shaping strategy used during training
        auxiliary_rewards : str
            Auxiliary reward strategy used during training
        """
        self.model_path = model_path
        self.reward_shaping = reward_shaping
        self.auxiliary_rewards = auxiliary_rewards
        self.model = None
        self.results = {}

        self._load_model()

    def _load_model(self):
        """Load trained model."""
        if os.path.exists(f"{self.model_path}.zip"):
            self.model = PPO.load(f"{self.model_path}.zip")
            print(f"Loaded model from {self.model_path}.zip")
        else:
            raise FileNotFoundError(f"Model not found: {self.model_path}.zip")

    def _create_env(self):
        """Create evaluation environment."""
        env = ChefsHatEnv(max_matches=3)
        if self.auxiliary_rewards:
            env = create_auxiliary_wrapper(self.auxiliary_rewards, env)
        if self.reward_shaping:
            env = create_reward_shaper(self.reward_shaping, env)
        return env

    def evaluate_performance(self, num_games: int = 100) -> Dict:
        """
        Evaluate agent performance over multiple games.

        Parameters
        ----------
        num_games : int
            Number of games to play

        Returns
        -------
        dict
            Performance metrics
        """
        env = self._create_env()
        rewards = []
        episode_lengths = []
        wins = 0
        losses = 0

        for game_idx in range(num_games):
            obs, info = env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            rewards.append(total_reward)
            episode_lengths.append(steps)

            if total_reward > 0:
                wins += 1
            else:
                losses += 1

            if (game_idx + 1) % 20 == 0:
                print(f"  Completed {game_idx + 1}/{num_games} games")

        env.close()

        win_rate = wins / num_games if num_games > 0 else 0

        self.results["performance"] = {
            "num_games": num_games,
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "median_reward": float(np.median(rewards)),
            "win_rate": float(win_rate),
            "wins": int(wins),
            "losses": int(losses),
            "avg_episode_length": float(np.mean(episode_lengths)),
            "std_episode_length": float(np.std(episode_lengths)),
        }

        return self.results["performance"]

    def evaluate_consistency(self, num_runs: int = 10, games_per_run: int = 20) -> Dict:
        """
        Evaluate performance consistency across multiple runs.

        Parameters
        ----------
        num_runs : int
            Number of independent evaluation runs
        games_per_run : int
            Games per run

        Returns
        -------
        dict
            Consistency metrics
        """
        run_rewards = []

        for run_idx in range(num_runs):
            env = self._create_env()
            rewards = []

            for game_idx in range(games_per_run):
                obs, info = env.reset()
                total_reward = 0
                done = False

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    done = terminated or truncated

                rewards.append(total_reward)

            run_rewards.append(np.mean(rewards))
            env.close()

        consistency_results = {
            "num_runs": num_runs,
            "avg_run_mean_reward": float(np.mean(run_rewards)),
            "std_run_mean_reward": float(np.std(run_rewards)),
            "cv_coefficient": float(
                np.std(run_rewards) / np.mean(run_rewards)
                if np.mean(run_rewards) != 0
                else 0
            ),
            "run_rewards": [float(r) for r in run_rewards],
        }

        self.results["consistency"] = consistency_results
        return consistency_results

    def summary(self) -> Dict:
        """Get evaluation summary."""
        return self.results


def evaluate_single_agent(
    model_dir: str,
    reward_shaping: str = "none",
    auxiliary_rewards: str = "none",
    num_games: int = 100,
):
    """Evaluate a single agent."""
    model_path = os.path.join(model_dir, "ppo_model")

    print(f"\n{'='*60}")
    print(f"Evaluating Agent")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Reward Shaping: {reward_shaping}")
    print(f"Auxiliary Rewards: {auxiliary_rewards}")
    print(f"{'='*60}\n")

    evaluator = AgentEvaluator(
        model_path,
        reward_shaping=reward_shaping,
        auxiliary_rewards=auxiliary_rewards,
    )

    print(f"Evaluating performance ({num_games} games)...")
    perf = evaluator.evaluate_performance(num_games=num_games)

    print(f"\nPerformance Results:")
    print(f"  Win Rate: {perf['win_rate']:.2%}")
    print(f"  Avg Reward: {perf['avg_reward']:.2f} ± {perf['std_reward']:.2f}")
    print(f"  Reward Range: [{perf['min_reward']:.1f}, {perf['max_reward']:.1f}]")
    print(f"  Avg Episode Length: {perf['avg_episode_length']:.1f} ± {perf['std_episode_length']:.1f}")

    print(f"\nEvaluating consistency...")
    cons = evaluator.evaluate_consistency(num_runs=10, games_per_run=20)

    print(f"\nConsistency Results:")
    print(f"  Mean of Run Means: {cons['avg_run_mean_reward']:.2f}")
    print(f"  Std of Run Means: {cons['std_run_mean_reward']:.2f}")
    print(f"  Coefficient of Variation: {cons['cv_coefficient']:.3f}")

    # Save results
    results_path = os.path.join(model_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(evaluator.summary(), f, indent=2)

    print(f"\nResults saved to {results_path}")

    return evaluator


def plot_comparison_results(experiment_dir: str):
    """
    Plot comparison results across experiments.

    Parameters
    ----------
    experiment_dir : str
        Directory containing experiment results
    """
    results = {}

    # Collect results from all experiments
    for exp_folder in os.listdir(experiment_dir):
        exp_path = os.path.join(experiment_dir, exp_folder)
        if not os.path.isdir(exp_path):
            continue

        metadata_path = os.path.join(exp_path, "training_metadata.json")
        eval_path = os.path.join(exp_path, "evaluation_results.json")

        if os.path.exists(metadata_path) and os.path.exists(eval_path):
            with open(metadata_path) as f:
                training = json.load(f)
            with open(eval_path) as f:
                evaluation = json.load(f)

            results[exp_folder] = {
                "training": training,
                "evaluation": evaluation,
            }

    if not results:
        print("No results found to plot")
        return

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Sparse/Delayed Reward Variant: Strategy Comparison", fontsize=14, fontweight="bold"
    )

    # Sort by experiment name
    sorted_exp = sorted(results.keys())

    # Plot 1: Win Rate
    ax = axes[0, 0]
    win_rates = [
        results[exp]["evaluation"]["performance"]["win_rate"] for exp in sorted_exp
    ]
    ax.bar(range(len(sorted_exp)), win_rates, color="steelblue", alpha=0.7)
    ax.set_ylabel("Win Rate", fontweight="bold")
    ax.set_title("Win Rate Comparison")
    ax.set_xticks(range(len(sorted_exp)))
    ax.set_xticklabels(sorted_exp, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: Average Reward
    ax = axes[0, 1]
    avg_rewards = [
        results[exp]["evaluation"]["performance"]["avg_reward"] for exp in sorted_exp
    ]
    std_rewards = [
        results[exp]["evaluation"]["performance"]["std_reward"] for exp in sorted_exp
    ]
    ax.bar(range(len(sorted_exp)), avg_rewards, yerr=std_rewards, capsize=5, color="seagreen", alpha=0.7)
    ax.set_ylabel("Average Reward", fontweight="bold")
    ax.set_title("Average Reward Comparison")
    ax.set_xticks(range(len(sorted_exp)))
    ax.set_xticklabels(sorted_exp, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Plot 3: Episode Length
    ax = axes[1, 0]
    episode_lengths = [
        results[exp]["evaluation"]["performance"]["avg_episode_length"] 
        for exp in sorted_exp
    ]
    ax.bar(range(len(sorted_exp)), episode_lengths, color="coral", alpha=0.7)
    ax.set_ylabel("Average Episode Length", fontweight="bold")
    ax.set_title("Episode Length Comparison")
    ax.set_xticks(range(len(sorted_exp)))
    ax.set_xticklabels(sorted_exp, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Plot 4: Consistency (Coefficient of Variation)
    ax = axes[1, 1]
    cvs = [
        results[exp]["evaluation"]["consistency"]["cv_coefficient"] for exp in sorted_exp
    ]
    ax.bar(range(len(sorted_exp)), cvs, color="mediumpurple", alpha=0.7)
    ax.set_ylabel("Coefficient of Variation", fontweight="bold")
    ax.set_title("Performance Consistency (Lower is Better)")
    ax.set_xticks(range(len(sorted_exp)))
    ax.set_xticklabels(sorted_exp, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    comparison_plot_path = os.path.join(experiment_dir, "comparison_results.png")
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to {comparison_plot_path}")
    plt.close()

    # Create detailed comparison table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("tight")
    ax.axis("off")

    table_data = []
    table_data.append([
        "Experiment",
        "Win Rate",
        "Avg Reward",
        "Std Reward",
        "Avg Length",
        "Consistency",
    ])

    for exp in sorted_exp:
        perf = results[exp]["evaluation"]["performance"]
        cons = results[exp]["evaluation"]["consistency"]
        table_data.append([
            exp,
            f"{perf['win_rate']:.1%}",
            f"{perf['avg_reward']:.2f}",
            f"{perf['std_reward']:.2f}",
            f"{perf['avg_episode_length']:.1f}",
            f"{cons['cv_coefficient']:.3f}",
        ])

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.15],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(table_data)):
        color = "#f0f0f0" if i % 2 == 0 else "white"
        for j in range(6):
            table[(i, j)].set_facecolor(color)

    plt.title("Detailed Comparison Results", fontweight="bold", fontsize=12, pad=20)
    table_path = os.path.join(experiment_dir, "comparison_table.png")
    plt.savefig(table_path, dpi=150, bbox_inches="tight")
    print(f"Comparison table saved to {table_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agents")
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to model directory",
    )
    parser.add_argument(
        "--reward-shaping",
        type=str,
        default="none",
        help="Reward shaping strategy",
    )
    parser.add_argument(
        "--auxiliary",
        type=str,
        default="none",
        help="Auxiliary reward strategy",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of games for evaluation",
    )
    parser.add_argument(
        "--plot-compare",
        action="store_true",
        help="Generate comparison plots for all experiments",
    )

    args = parser.parse_args()

    if args.plot_compare:
        plot_comparison_results(args.model_dir)
    else:
        evaluate_single_agent(
            args.model_dir,
            reward_shaping=args.reward_shaping,
            auxiliary_rewards=args.auxiliary,
            num_games=args.num_games,
        )


if __name__ == "__main__":
    main()
