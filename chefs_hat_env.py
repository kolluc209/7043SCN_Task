"""
ChefsHatEnv - A Gymnasium-compatible wrapper for the Chef's Hat card game.

The RL agent controls player 0. The other 3 players act randomly.
Observation: float32 vector of length 28 (hand[17] + board[11])
Action space: Discrete(200) — the 200 high-level actions defined by the game
"""

import sys
import os
import random
import logging

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add src to path so we can import the game modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from core.game_env.game import Game
from core.utils.rules import get_high_level_actions, get_possible_actions, complement_array
from core.utils.cards import deal_cards


# Suppress the noisy engine logger during training
logging.getLogger().setLevel(logging.WARNING)


class SilentLogger:
    """A no-op logger that satisfies the engine's logging interface."""
    def engine_log(self, *args, **kwargs):
        pass
    def info(self, *args, **kwargs):
        pass
    def debug(self, *args, **kwargs):
        pass


class ChefsHatEnv(gym.Env):
    """
    Gymnasium environment for Chef's Hat card game.

    The RL agent is player 0. Players 1-3 play randomly.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_matches=1, max_rounds=None):
        super().__init__()

        self.all_actions = get_high_level_actions()  # 200 actions
        self.num_actions = len(self.all_actions)

        # Observation: hand (17 floats) + board (11 floats) = 28
        self.observation_space = spaces.Box(
            low=0.0, high=13.0, shape=(28,), dtype=np.float32
        )
        # Action: index into the all_actions list
        self.action_space = spaces.Discrete(self.num_actions)

        self.max_matches = max_matches
        self.max_rounds = max_rounds

        self.agent_name = "RLAgent"
        self.opponent_names = ["Opponent_1", "Opponent_2", "Opponent_3"]
        self.all_player_names = [self.agent_name] + self.opponent_names

        self.logger = SilentLogger()
        self.game = None

    def _get_obs_from_observation(self, obs_dict):
        """Convert game observation dict to flat numpy array."""
        hand = np.array(obs_dict["hand"], dtype=np.float32)
        board = np.array(obs_dict["board"], dtype=np.float32)
        return np.concatenate([hand, board])

    def _random_action_for(self, possible_actions):
        """Pick a random action from the list of possible actions (for opponents)."""
        non_pass = [a for a in possible_actions if a != "pass"]
        chosen = random.choice(non_pass if non_pass else possible_actions)
        return chosen

    def _advance_until_agent_turn_or_done(self):
        """
        Keep stepping the game (playing for opponents) until
        it's the RL agent's turn to act, or the game ends.

        Returns (observation, reward, terminated, truncated, info)
        """
        while True:
            # Request action from current player
            result = self.game.step(None)

            if result is None:
                # Game is completely finished
                return self._make_final_obs(), self._compute_reward(), True, False, {"game_over": True}

            if result.get("request_action"):
                player_name = result["player"]

                if player_name == self.agent_name:
                    # It's the RL agent's turn
                    obs = self._get_obs_from_observation(result["observation"])
                    self._current_possible_actions = result["observation"]["possible_actions"]
                    return obs, 0.0, False, False, {"possible_actions": self._current_possible_actions}

                else:
                    # It's an opponent's turn — play randomly
                    possible = result["observation"]["possible_actions"]
                    action_str = self._random_action_for(possible)
                    step_result = self.game.step(action_str)

                    if step_result is None or self.game.finished:
                        return self._make_final_obs(), self._compute_reward(), True, False, {"game_over": True}

                    # Check for match_over
                    if step_result.get("match_over"):
                        if self.game.finished:
                            return self._make_final_obs(), self._compute_reward(), True, False, {"game_over": True}
                        else:
                            # Start next match
                            self._start_new_match()
                            continue
                    continue

    def _make_final_obs(self):
        """Create an observation when game is over."""
        return np.zeros(28, dtype=np.float32)

    def _compute_reward(self):
        """Compute reward based on final game scores."""
        if self.game is None:
            return 0.0
        scores = self.game.scores
        agent_score = scores.get(self.agent_name, 0)
        max_opponent_score = max(scores.get(n, 0) for n in self.opponent_names)

        # Reward: agent score minus best opponent score
        return float(agent_score - max_opponent_score)

    def _start_new_match(self):
        """Deal cards and start a new match."""
        self.game.deal_cards()
        self.game.create_new_match()
        self.game.start_match()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game = Game(
            player_names=self.all_player_names,
            max_matches=self.max_matches,
            max_rounds=self.max_rounds,
            logger=self.logger,
            save_dataset=False,
            dataset_directory="",
        )
        self.game.start()
        self._start_new_match()
        self._current_possible_actions = []

        obs, reward, terminated, truncated, info = self._advance_until_agent_turn_or_done()
        return obs, info

    def step(self, action):
        """
        Take an action as the RL agent.

        Parameters
        ----------
        action : int
            Index into the all_actions list.
        """
        action_str = self.all_actions[action]

        # Validate the action
        if action_str not in self._current_possible_actions:
            # Invalid action: penalize and pick a random valid one
            if self._current_possible_actions:
                action_str = random.choice(self._current_possible_actions)
            reward_penalty = -1.0
        else:
            reward_penalty = 0.0

        # Execute the action
        result = self.game.step(action_str)

        if result is None or self.game.finished:
            obs = self._make_final_obs()
            reward = self._compute_reward() + reward_penalty
            return obs, reward, True, False, {"game_over": True}

        # Check for match_over
        if result.get("match_over"):
            if self.game.finished:
                obs = self._make_final_obs()
                reward = self._compute_reward() + reward_penalty
                return obs, reward, True, False, {"game_over": True}
            else:
                self._start_new_match()

        # Advance game until it's the agent's turn again (or game ends)
        obs, step_reward, terminated, truncated, info = self._advance_until_agent_turn_or_done()
        reward = step_reward + reward_penalty

        return obs, reward, terminated, truncated, info

    def get_action_mask(self):
        """
        Return a boolean mask of valid actions.
        Useful for masked PPO or action filtering.
        """
        mask = np.zeros(self.num_actions, dtype=np.bool_)
        for action_str in self._current_possible_actions:
            if action_str in self.all_actions:
                idx = self.all_actions.index(action_str)
                mask[idx] = True
        return mask
