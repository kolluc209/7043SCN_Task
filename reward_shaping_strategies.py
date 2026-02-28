"""
Reward Shaping Strategies for Chef's Hat Gym

This module implements multiple reward shaping approaches for handling
sparse, delayed rewards in the Chef's Hat card game. The focus is on
the Sparse/Delayed Reward Variant, exploring different techniques to
provide intermediate learning signals.

Author: Students (Sparse/Delayed Reward Variant - ID mod 7 = 2)
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any


class BaseRewardShaper(gym.Wrapper):
    """Base class for reward shaping strategies."""

    def __init__(self, env):
        super().__init__(env)
        self.episode_step = 0
        self.match_started = False

    def reset(self, seed=None, options=None):
        self.episode_step = 0
        self.match_started = False
        return self.env.reset(seed=seed, options=options)

    def _shape_reward(self, obs, reward, terminated, truncated, info):
        """Override in subclass to implement shaping."""
        return reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1

        shaped_reward = self._shape_reward(obs, reward, terminated, truncated, info)
        return obs, shaped_reward, terminated, truncated, info


class HandSizeRewardShaper(BaseRewardShaper):
    """
    Reward shaping based on hand size reduction.
    
    Rationale: Playing cards reduces hand size, which is a sign of progress.
    This provides intermediate learning signals during sparse reward periods.
    """

    def __init__(self, env, hand_reduction_bonus=0.05, invalid_action_penalty=-0.5):
        super().__init__(env)
        self.hand_reduction_bonus = hand_reduction_bonus
        self.invalid_action_penalty = invalid_action_penalty
        self.previous_hand_size = 17  # Max hand size in Chef's Hat
        self.step_counter = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Extract hand size from observation (first 17 floats)
        self.previous_hand_size = np.sum(obs[:17])
        return obs, info

    def _shape_reward(self, obs, reward, terminated, truncated, info):
        """Shape reward based on hand size changes."""
        shaped_reward = reward

        if not (terminated or truncated):
            # Calculate current hand size from observation
            current_hand_size = np.sum(obs[:17])

            # Reward for reducing hand size (playing cards)
            if current_hand_size < self.previous_hand_size:
                hand_reduction = self.previous_hand_size - current_hand_size
                shaped_reward += self.hand_reduction_bonus * hand_reduction

            self.previous_hand_size = current_hand_size

        # Penalty for invalid actions
        if "possible_actions" in info:
            # Small penalty is already applied in env, no need to double
            pass

        # Bonus for final positive outcome
        if (terminated or truncated) and reward > 0:
            shaped_reward += 5.0

        # Penalty for final negative outcome
        if (terminated or truncated) and reward < 0:
            shaped_reward -= 5.0

        return shaped_reward


class ProgressRewardShaper(BaseRewardShaper):
    """
    Reward shaping based on progress indicators.
    
    Rationale: Encourages the agent to take actions that change the game state,
    promoting exploration and reducing stagnation.
    """

    def __init__(self, env, progress_bonus=0.02, stagnation_penalty=-0.01):
        super().__init__(env)
        self.progress_bonus = progress_bonus
        self.stagnation_penalty = stagnation_penalty
        self.previous_obs = None
        self.idle_counter = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.previous_obs = obs.copy()
        self.idle_counter = 0
        return obs, info

    def _shape_reward(self, obs, reward, terminated, truncated, info):
        """Shape reward based on game state changes."""
        shaped_reward = reward

        if not (terminated or truncated):
            # Detect if state changed (progress)
            obs_change = np.sum(np.abs(obs - self.previous_obs))

            if obs_change > 0:
                shaped_reward += self.progress_bonus
                self.idle_counter = 0
            else:
                # Penalize idle states
                self.idle_counter += 1
                if self.idle_counter > 3:
                    shaped_reward += self.stagnation_penalty

            self.previous_obs = obs.copy()

        # Large bonus/penalty at episode end
        if terminated or truncated:
            if reward > 0:
                shaped_reward += 10.0
            else:
                shaped_reward -= 10.0

        return shaped_reward


class HybridRewardShaper(BaseRewardShaper):
    """
    Hybrid reward shaping combining multiple signals.
    
    Rationale: Combines hand size reduction and game state changes
    for a more comprehensive learning signal.
    """

    def __init__(
        self,
        env,
        hand_reduction_weight=0.05,
        state_change_weight=0.02,
        endpoint_weight=10.0,
    ):
        super().__init__(env)
        self.hand_reduction_weight = hand_reduction_weight
        self.state_change_weight = state_change_weight
        self.endpoint_weight = endpoint_weight
        self.previous_hand_size = 17
        self.previous_obs = None

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.previous_hand_size = np.sum(obs[:17])
        self.previous_obs = obs.copy()
        return obs, info

    def _shape_reward(self, obs, reward, terminated, truncated, info):
        """Combine multiple reward shaping signals."""
        shaped_reward = reward

        if not (terminated or truncated):
            # Hand size reduction signal
            current_hand_size = np.sum(obs[:17])
            if current_hand_size < self.previous_hand_size:
                hand_reduction = self.previous_hand_size - current_hand_size
                shaped_reward += self.hand_reduction_weight * hand_reduction

            # State change signal
            if self.previous_obs is not None:
                obs_change = np.sum(np.abs(obs - self.previous_obs))
                if obs_change > 0:
                    shaped_reward += self.state_change_weight

            self.previous_hand_size = current_hand_size
            self.previous_obs = obs.copy()
        else:
            # Strong endpoint signal
            if reward > 0:
                shaped_reward += self.endpoint_weight
            else:
                shaped_reward -= self.endpoint_weight

        return shaped_reward


class CurriculumRewardShaper(BaseRewardShaper):
    """
    Curriculum-based reward shaping that evolves over time.
    
    Rationale: Start with dense intermediate signals and gradually
    transition to sparse rewards as training progresses.
    """

    def __init__(
        self,
        env,
        total_episodes=50000,
        initial_bonus=0.1,
        final_bonus=0.01,
    ):
        super().__init__(env)
        self.total_episodes = total_episodes
        self.initial_bonus = initial_bonus
        self.final_bonus = final_bonus
        self.current_episode = 0
        self.previous_hand_size = 17

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.previous_hand_size = np.sum(obs[:17])
        self.current_episode += 1
        return obs, info

    def _get_curriculum_bonus(self):
        """Interpolate bonus based on training progress."""
        progress = min(1.0, self.current_episode / self.total_episodes)
        return self.initial_bonus - (self.initial_bonus - self.final_bonus) * progress

    def _shape_reward(self, obs, reward, terminated, truncated, info):
        """Apply curriculum-based shaping."""
        shaped_reward = reward
        bonus = self._get_curriculum_bonus()

        if not (terminated or truncated):
            current_hand_size = np.sum(obs[:17])
            if current_hand_size < self.previous_hand_size:
                hand_reduction = self.previous_hand_size - current_hand_size
                shaped_reward += bonus * hand_reduction
            self.previous_hand_size = current_hand_size
        else:
            if reward > 0:
                shaped_reward += 5.0
            else:
                shaped_reward -= 5.0

        return shaped_reward


class NoRewardShaper(BaseRewardShaper):
    """Baseline: no reward shaping, just raw rewards."""

    def _shape_reward(self, obs, reward, terminated, truncated, info):
        """Return raw reward without modification."""
        return reward


def create_reward_shaper(strategy: str, env, **kwargs) -> gym.Env:
    """
    Factory function to create a reward shaper.

    Parameters
    ----------
    strategy : str
        One of: 'none', 'hand_size', 'progress', 'hybrid', 'curriculum'
    env : gym.Env
        The base environment
    **kwargs : dict
        Additional arguments passed to the shaper

    Returns
    -------
    gym.Env
        Environment wrapped with reward shaping
    """
    shapers = {
        "none": NoRewardShaper,
        "hand_size": HandSizeRewardShaper,
        "progress": ProgressRewardShaper,
        "hybrid": HybridRewardShaper,
        "curriculum": CurriculumRewardShaper,
    }

    if strategy not in shapers:
        raise ValueError(f"Unknown reward shaping strategy: {strategy}")

    return shapers[strategy](env, **kwargs)
