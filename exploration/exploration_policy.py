"""Defines exploration policies for recommendation diversity."""

import random
from typing import List, Dict, Any
from enum import Enum


class ExplorationStrategy(str, Enum):
    UCB = "ucb"
    EPSILON_GREEDY = "epsilon_greedy"
    THOMPSON_SAMPLING = "thompson_sampling"
    RANDOM = "random"


class ExplorationPolicy:
    """
    Manages exploration/exploitation trade-off in recommendation.
    Supports multiple strategies configurable per experiment.
    """

    def __init__(self, strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY,
                 epsilon: float = 0.1,
                 exploration_fraction: float = 0.1):
        self.strategy = strategy
        self.epsilon = epsilon
        self.exploration_fraction = exploration_fraction

    def apply(self, exploited: List[Dict[str, Any]],
              exploration_pool: List[Dict[str, Any]],
              final_k: int = 20) -> List[Dict[str, Any]]:
        """
        Mix exploited (ranked) and exploration candidates.
        Returns a list of final_k recommendations with exploration items injected.
        """
        n_explore = max(1, int(final_k * self.exploration_fraction))
        n_exploit = final_k - n_explore

        exploit_items = exploited[:n_exploit]

        if self.strategy == ExplorationStrategy.RANDOM:
            explore_items = random.sample(exploration_pool, min(n_explore, len(exploration_pool)))
        else:
            explore_items = exploration_pool[:n_explore]

        # Interleave exploration items at random positions
        result = exploit_items.copy()
        positions = sorted(random.sample(range(final_k), min(n_explore, final_k)))
        for i, item in enumerate(explore_items):
            item["is_exploration"] = True
            pos = positions[i] if i < len(positions) else len(result)
            result.insert(min(pos, len(result)), item)

        return result[:final_k]
