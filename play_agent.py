"""
Play and Visualize Trained Agents

This script loads a trained agent and demonstrates its gameplay,
showing decision-making and performance metrics in real-time.

Usage:
    python play_agent.py path/to/model --episodes 5 --verbose
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from chefs_hat_env import ChefsHatEnv
from reward_shaping_strategies import create_reward_shaper
from auxiliary_rewards import create_auxiliary_wrapper


class AgentPlayer:
    """Plays trained agents and provides visualization/logging."""

    def __init__(
        self,
        model_path: str,
        reward_shaping: str = "none",
        auxiliary_rewards: str = "none",
        verbose: bool = True,
    ):
        """Initialize player."""
        self.model_path = model_path
        self.reward_shaping = reward_shaping
        self.auxiliary_rewards = auxiliary_rewards
        self.verbose = verbose

        # Load model
        if os.path.exists(f"{model_path}.zip"):
            self.model = PPO.load(f"{model_path}.zip")
            self.log(f"Loaded model from {model_path}.zip")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}.zip")

    def log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)

    def create_env(self):
        """Create evaluation environment."""
        env = ChefsHatEnv(max_matches=3)
        if self.auxiliary_rewards:
            env = create_auxiliary_wrapper(self.auxiliary_rewards, env)
        if self.reward_shaping:
            env = create_reward_shaper(self.reward_shaping, env)
        return env

    def play_episode(self, episode_num: int = 1, deterministic: bool = True):
        """
        Play one full episode.

        Parameters
        ----------
        episode_num : int
            Episode number for logging
        deterministic : bool
            If True, use greedy policy; else sample

        Returns
        -------
        dict
            Episode statistics
        """
        env = self.create_env()
        obs, info = env.reset()

        total_reward = 0
        step_count = 0
        action_log = []

        self.log(f"\n{'='*70}")
        self.log(f"Episode {episode_num}")
        self.log(f"{'='*70}")
        self.log(f"Initial board state:")
        self.log(f"  - Hand cards: {obs[:17].sum():.0f}")
        self.log(f"  - Possible actions: {len(info['possible_actions'])}")

        done = False
        while not done:
            # Get action from model
            action, value = self.model.predict(obs, deterministic=deterministic)
            action_str = env.all_actions[action]

            # Check if action is valid
            is_valid = action_str in info.get("possible_actions", [])

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            step_count += 1

            action_log.append(
                {
                    "step": step_count,
                    "action": action_str,
                    "valid": is_valid,
                    "reward": reward,
                    "hand_size": obs[:17].sum(),
                }
            )

            if step_count % 10 == 0 or done:
                hand_size = obs[:17].sum()
                self.log(
                    f"  Step {step_count:>3d}: Action='{action_str[:20]:20s}' "
                    f"Hand={hand_size:>2.0f} Reward={reward:>7.2f}"
                )

        env.close()

        result = {
            "episode": episode_num,
            "total_reward": total_reward,
            "steps": step_count,
            "actions": action_log,
            "outcome": "WIN" if total_reward > 0 else "LOSS",
            "margin": total_reward,
        }

        self.log(f"\n{'─'*70}")
        self.log(f"Episode Result:")
        self.log(f"  - Outcome: {result['outcome']}")
        self.log(f"  - Final Reward: {total_reward:>7.2f}")
        self.log(f"  - Steps: {step_count}")
        self.log(f"  - Valid Actions: {sum(1 for a in action_log if a['valid'])}/{len(action_log)}")
        self.log(f"{'='*70}")

        return result

    def play_multiple(self, num_episodes: int = 5, deterministic: bool = True):
        """
        Play multiple episodes and aggregate statistics.

        Parameters
        ----------
        num_episodes : int
            Number of episodes to play
        deterministic : bool
            If True, use greedy policy

        Returns
        -------
        dict
            Aggregated statistics
        """
        results = []
        wins = 0
        total_rewards = []

        self.log(f"\n\n{'#'*70}")
        self.log(f"# Playing {num_episodes} Episodes")
        self.log(f"#{'─'*68}#")
        self.log(f"# Mode: {'Deterministic (Greedy)' if deterministic else 'Stochastic'}")
        self.log(f"{'#'*70}\n")

        for ep_num in range(1, num_episodes + 1):
            result = self.play_episode(episode_num=ep_num, deterministic=deterministic)
            results.append(result)

            if result["outcome"] == "WIN":
                wins += 1

            total_rewards.append(result["total_reward"])

        # Aggregate statistics
        stats = {
            "num_episodes": num_episodes,
            "wins": wins,
            "losses": num_episodes - wins,
            "win_rate": wins / num_episodes if num_episodes > 0 else 0,
            "avg_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "max_reward": np.max(total_rewards),
            "min_reward": np.min(total_rewards),
            "median_reward": np.median(total_rewards),
            "episodes": results,
        }

        # Print summary
        self.log(f"\n\n{'='*70}")
        self.log(f"SUMMARY - {num_episodes} Episodes")
        self.log(f"{'='*70}")
        self.log(f"Win Rate:        {wins}/{num_episodes} ({100*stats['win_rate']:.1f}%)")
        self.log(f"Average Reward:  {stats['avg_reward']:>7.2f}")
        self.log(f"Std Dev:         {stats['std_reward']:>7.2f}")
        self.log(f"Reward Range:    [{stats['min_reward']:>7.2f}, {stats['max_reward']:>7.2f}]")
        self.log(f"{'='*70}\n")

        return stats

    def play_interactive(self):
        """Play a single episode with detailed step-by-step output."""
        self.play_episode(deterministic=True)


def main():
    parser = argparse.ArgumentParser(description="Play with trained agents")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model (without .zip)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to play",
    )
    parser.add_argument(
        "--reward-shaping",
        type=str,
        default="none",
        help="Reward shaping strategy used during training",
    )
    parser.add_argument(
        "--auxiliary",
        type=str,
        default="none",
        help="Auxiliary reward strategy used during training",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic (sampling) policy instead of deterministic (greedy)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()

    try:
        player = AgentPlayer(
            args.model_path,
            reward_shaping=args.reward_shaping,
            auxiliary_rewards=args.auxiliary,
            verbose=not args.quiet,
        )

        stats = player.play_multiple(
            num_episodes=args.episodes,
            deterministic=not args.stochastic,
        )

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nUsage: python play_agent.py path/to/model --episodes 5")
        sys.exit(1)
    except Exception as e:
        print(f"Error during playback: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
