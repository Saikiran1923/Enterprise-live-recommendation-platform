"""Contextual bandit for exploration in recommendations."""

import math
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class UCBContextualBandit:
    """
    Upper Confidence Bound (UCB1) contextual bandit.
    Balances exploration of new content with exploitation of known preferences.
    """

    def __init__(self, exploration_coeff: float = 1.0,
                 min_observations: int = 5):
        self.exploration_coeff = exploration_coeff
        self.min_observations = min_observations
        self._arm_counts: Dict[str, int] = defaultdict(int)
        self._arm_rewards: Dict[str, float] = defaultdict(float)
        self._total_pulls = 0

    def select(self, candidates: List[Dict[str, Any]],
               context: Dict[str, Any],
               n_explore: int = 2) -> List[Dict[str, Any]]:
        """
        Select exploration candidates using UCB1.
        Returns up to n_explore candidates prioritizing under-explored items.
        """
        unexplored = [
            c for c in candidates
            if self._arm_counts[c["video_id"]] < self.min_observations
        ]
        if unexplored:
            return unexplored[:n_explore]

        ucb_scores = []
        for c in candidates:
            vid = c["video_id"]
            n = max(self._arm_counts[vid], 1)
            mean_reward = self._arm_rewards[vid] / n
            confidence = self.exploration_coeff * math.sqrt(
                math.log(max(self._total_pulls, 1)) / n
            )
            ucb_scores.append((c, mean_reward + confidence))

        ucb_scores.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in ucb_scores[:n_explore]]

    def update(self, video_id: str, reward: float) -> None:
        """Update arm statistics with observed reward."""
        self._arm_counts[video_id] += 1
        self._arm_rewards[video_id] += reward
        self._total_pulls += 1

    def get_arm_stats(self, video_id: str) -> Dict[str, Any]:
        n = self._arm_counts[video_id]
        return {
            "pulls": n,
            "mean_reward": self._arm_rewards[video_id] / max(n, 1),
            "is_cold": n < self.min_observations,
        }
