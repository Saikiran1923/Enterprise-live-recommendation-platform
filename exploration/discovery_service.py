"""Discovery service for surfacing new and diverse content."""

import logging
from typing import List, Dict, Any, Optional

from exploration.contextual_bandit import UCBContextualBandit
from exploration.exploration_policy import ExplorationPolicy, ExplorationStrategy

logger = logging.getLogger(__name__)


class DiscoveryService:
    """
    Manages content discovery and exploration to prevent filter bubbles
    and help users find new interests.
    """

    def __init__(self, exploration_fraction: float = 0.1):
        self._bandit = UCBContextualBandit()
        self._policy = ExplorationPolicy(
            strategy=ExplorationStrategy.EPSILON_GREEDY,
            exploration_fraction=exploration_fraction,
        )

    async def get_discovery_candidates(self,
                                        user_id: str,
                                        current_candidates: List[Dict[str, Any]],
                                        context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select exploration candidates via bandit."""
        return self._bandit.select(current_candidates, context)

    def inject_exploration(self, ranked: List[Dict[str, Any]],
                           exploration_pool: List[Dict[str, Any]],
                           final_k: int = 20) -> List[Dict[str, Any]]:
        """Mix ranked and exploration items into final recommendation list."""
        return self._policy.apply(ranked, exploration_pool, final_k)

    def record_feedback(self, video_id: str, watch_rate: float,
                        liked: bool = False) -> None:
        """Update bandit with observed engagement feedback."""
        reward = watch_rate * 0.7 + float(liked) * 0.3
        self._bandit.update(video_id, reward)
