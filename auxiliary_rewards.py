"""
Auxiliary Reward and Credit Assignment Mechanisms

This module implements auxiliary rewards and advanced credit assignment
strategies to improve learning from sparse, delayed rewards.

Techniques include:
- N-step returns for better credit assignment
- Intrinsic motivation (curiosity, empowerment)
- Trajectory-based rewards
- Value-based auxiliary tasks

Author: Students (Sparse/Delayed Reward Variant - ID mod 7 = 2)
"""

import numpy as np
import gymnasium as gym
from collections import deque
from typing import List, Tuple, Dict, Any


class AuxiliaryRewardBuffer:
    """Stores and processes trajectory data for auxiliary rewards."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.observations = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.hand_sizes = deque(maxlen=max_size)
        self.terminateds = deque(maxlen=max_size)

    def add_transition(self, obs, action, reward, hand_size, terminated):
        """Add a transition to the buffer."""
        self.observations.append(obs.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.hand_sizes.append(hand_size)
        self.terminateds.append(terminated)

    def compute_nstep_returns(self, gamma: float = 0.99, n: int = 3):
        """
        Compute n-step returns for better credit assignment.
        
        Parameters
        ----------
        gamma : float
            Discount factor
        n : int
            Number of steps for lookahead
            
        Returns
        -------
        np.array
            N-step returns for each transition
        """
        returns = np.zeros(len(self.rewards))
        trajectory_len = len(self.rewards)

        for i in range(trajectory_len):
            cumulative_reward = 0
            discount = 1.0

            for j in range(n):
                if i + j < trajectory_len:
                    cumulative_reward += discount * self.rewards[i + j]
                    if self.terminateds[i + j]:
                        break
                    discount *= gamma

            returns[i] = cumulative_reward

        return returns

    def compute_lambda_returns(self, gamma: float = 0.99, lambda_gae: float = 0.95):
        """
        Compute lambda-return (GAE-style) for credit assignment.
        
        Uses a weighted average of n-step returns.
        """
        returns = np.zeros(len(self.rewards))
        trajectory_len = len(self.rewards)

        for i in range(trajectory_len):
            cumulative_return = 0
            lambda_power = 1.0
            discount = 1.0

            for n in range(1, trajectory_len - i + 1):
                n_step_return = self._compute_single_nstep(i, n, gamma)
                cumulative_return += (1 - lambda_gae) * lambda_power * n_step_return
                lambda_power *= lambda_gae
                discount *= gamma

            returns[i] = cumulative_return

        return returns

    def _compute_single_nstep(self, start_idx: int, n: int, gamma: float):
        """Compute single n-step return."""
        cumulative = 0
        discount = 1.0
        for j in range(n):
            if start_idx + j < len(self.rewards):
                cumulative += discount * self.rewards[start_idx + j]
                if self.terminateds[start_idx + j]:
                    break
                discount *= gamma
        return cumulative

    def clear(self):
        """Clear all buffers."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.hand_sizes.clear()
        self.terminateds.clear()


class CuriosityRewardWrapper(gym.Wrapper):
    """
    Intrinsic motivation through curiosity (empowerment).
    
    Rewards the agent for visiting previously unseen or rarely-visited states.
    """

    def __init__(self, env, curiosity_bonus: float = 0.01, decay: float = 0.9999):
        super().__init__(env)
        self.curiosity_bonus = curiosity_bonus
        self.decay = decay
        self.state_visitation = {}
        self.visit_decay = 1.0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.visit_decay = 1.0
        return obs, info

    def _discretize_obs(self, obs: np.ndarray, bins: int = 10) -> tuple:
        """Discretize continuous observation for visitation tracking."""
        discretized = np.digitize(obs, bins=np.linspace(0, 13, bins))
        return tuple(discretized)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track state visitation
        state_key = self._discretize_obs(obs)
        if state_key not in self.state_visitation:
            self.state_visitation[state_key] = 0
        self.state_visitation[state_key] += 1

        # Compute curiosity bonus (inversely proportional to visitation count)
        curiosity_signal = self.curiosity_bonus / (1 + self.state_visitation[state_key])
        reward += curiosity_signal

        # Decay the bonus over time
        self.visit_decay *= self.decay
        self.curiosity_bonus *= self.visit_decay

        return obs, reward, terminated, truncated, info


class EntropyRewardWrapper(gym.Wrapper):
    """
    Intrinsic motivation through entropy maximization.
    
    Encourages diverse action exploration by rewarding state entropy.
    """

    def __init__(self, env, entropy_bonus: float = 0.01):
        super().__init__(env)
        self.entropy_bonus = entropy_bonus
        self.previous_obs = None

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.previous_obs = obs.copy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Compute entropy as a measure of state diversity
        if self.previous_obs is not None:
            obs_change = np.abs(obs - self.previous_obs)
            entropy_signal = np.sum(obs_change) * self.entropy_bonus / obs.shape[0]
            reward += entropy_signal

        self.previous_obs = obs.copy()
        return obs, reward, terminated, truncated, info


class TrajectoryRewardAugmentation(gym.Wrapper):
    """
    Augment rewards with trajectory-based signals.
    
    Provides intermediate rewards based on trajectory properties:
    - Card playing frequency
    - Hand size trajectory
    - Action diversity
    """

    def __init__(self, env, trajectory_window: int = 10):
        super().__init__(env)
        self.trajectory_window = trajectory_window
        self.trajectory_buffer = AuxiliaryRewardBuffer(max_size=trajectory_window)
        self.step_counter = 0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.trajectory_buffer.clear()
        self.step_counter = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track hand size for auxiliary reward
        hand_size = np.sum(obs[:17])

        # Add to buffer
        self.trajectory_buffer.add_transition(
            obs, action, reward, hand_size, terminated or truncated
        )

        # Compute trajectory-based auxiliary reward
        auxiliary_reward = self._compute_trajectory_reward()
        reward += auxiliary_reward

        self.step_counter += 1

        if terminated or truncated:
            self.trajectory_buffer.clear()

        return obs, reward, terminated, truncated, info

    def _compute_trajectory_reward(self) -> float:
        """Compute auxiliary reward based on trajectory properties."""
        if len(self.trajectory_buffer.hand_sizes) < 2:
            return 0.0

        hand_sizes = np.array(list(self.trajectory_buffer.hand_sizes))
        
        # Reward for consistent hand size reduction
        hand_reductions = np.maximum(
            np.diff(hand_sizes) * -1, 0
        )  # Positive where hand shrinks
        card_playing_reward = np.sum(hand_reductions) * 0.01

        return card_playing_reward


class MultiTaskAuxiliaryWrapper(gym.Wrapper):
    """
    Multi-task auxiliary learning for better credit assignment.
    
    Combines multiple auxiliary tasks:
    1. Game outcome prediction
    2. Hand size estimation
    3. State change detection
    """

    def __init__(self, env, aux_weight: float = 0.1):
        super().__init__(env)
        self.aux_weight = aux_weight
        self.buffer = AuxiliaryRewardBuffer(max_size=1000)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.buffer.clear()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        hand_size = np.sum(obs[:17])
        self.buffer.add_transition(obs, action, reward, hand_size, terminated or truncated)

        # Compute multi-task auxiliary rewards
        aux_reward = self._compute_auxiliary_signals(obs, terminated or truncated)
        reward += aux_reward * self.aux_weight

        if terminated or truncated:
            self.buffer.clear()

        return obs, reward, terminated, truncated, info

    def _compute_auxiliary_signals(self, obs: np.ndarray, done: bool) -> float:
        """Compute combined auxiliary reward signals."""
        auxiliary_total = 0.0

        # Task 1: Hand size change detection
        if len(self.buffer.hand_sizes) >= 2:
            prev_hand = self.buffer.hand_sizes[-2]
            curr_hand = np.sum(obs[:17])
            if curr_hand < prev_hand:
                auxiliary_total += (prev_hand - curr_hand) * 0.5

        # Task 2: State diversity
        if len(self.buffer.observations) >= 2:
            obs_change = np.sum(
                np.abs(
                    np.array(self.buffer.observations[-1])
                    - np.array(self.buffer.observations[-2])
                )
            )
            auxiliary_total += min(obs_change * 0.1, 1.0)

        # Task 3: Episode completion signal
        if done and len(self.buffer.rewards) > 0:
            final_reward = self.buffer.rewards[-1]
            if final_reward > 0:
                auxiliary_total += 2.0
            else:
                auxiliary_total -= 1.0

        return auxiliary_total


def create_auxiliary_wrapper(strategy: str, env, **kwargs) -> gym.Env:
    """
    Factory function to create an auxiliary reward wrapper.

    Parameters
    ----------
    strategy : str
        One of: 'none', 'curiosity', 'entropy', 'trajectory', 'multitask'
    env : gym.Env
        The base environment
    **kwargs : dict
        Additional arguments passed to the wrapper

    Returns
    -------
    gym.Env
        Environment wrapped with auxiliary rewards
    """
    wrappers = {
        "curiosity": CuriosityRewardWrapper,
        "entropy": EntropyRewardWrapper,
        "trajectory": TrajectoryRewardAugmentation,
        "multitask": MultiTaskAuxiliaryWrapper,
    }

    if strategy == "none" or strategy is None:
        return env

    if strategy not in wrappers:
        raise ValueError(f"Unknown auxiliary reward strategy: {strategy}")

    return wrappers[strategy](env, **kwargs)
