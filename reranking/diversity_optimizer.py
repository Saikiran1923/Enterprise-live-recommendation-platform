"""Diversity optimization for final recommendation list."""

import logging
from typing import List, Dict, Any, Set

logger = logging.getLogger(__name__)


class DiversityOptimizer:
    """
    Re-orders ranked candidates to maximize diversity while
    preserving overall relevance. Uses Maximum Marginal Relevance (MMR).
    """

    def __init__(self, diversity_weight: float = 0.3,
                 max_same_creator: int = 3,
                 max_same_category: int = 5):
        self.diversity_weight = diversity_weight
        self.max_same_creator = max_same_creator
        self.max_same_category = max_same_category

    def optimize(self, ranked_candidates: List[Dict[str, Any]],
                 top_k: int = 20) -> List[Dict[str, Any]]:
        """Apply diversity optimization to return a diverse top-k list."""
        if len(ranked_candidates) <= top_k:
            return ranked_candidates

        selected: List[Dict[str, Any]] = []
        creator_count: Dict[str, int] = {}
        category_count: Dict[str, int] = {}
        remaining = ranked_candidates.copy()

        while len(selected) < top_k and remaining:
            best_idx = self._pick_best(remaining, selected,
                                       creator_count, category_count)
            best = remaining.pop(best_idx)
            selected.append(best)

            creator = best.get("creator_id", "unknown")
            category = best.get("category", "unknown")
            creator_count[creator] = creator_count.get(creator, 0) + 1
            category_count[category] = category_count.get(category, 0) + 1

        return selected

    def _pick_best(self, remaining: List[Dict], selected: List[Dict],
                   creator_count: Dict, category_count: Dict) -> int:
        """Pick index of best next item balancing relevance and diversity."""
        best_idx = 0
        best_score = float("-inf")

        for i, cand in enumerate(remaining):
            creator = cand.get("creator_id", "unknown")
            category = cand.get("category", "unknown")

            # Hard constraints
            if creator_count.get(creator, 0) >= self.max_same_creator:
                continue
            if category_count.get(category, 0) >= self.max_same_category:
                continue

            relevance = cand.get("ranking_score", 0)
            diversity_penalty = (
                creator_count.get(creator, 0) * 0.1 +
                category_count.get(category, 0) * 0.05
            )
            score = (1 - self.diversity_weight) * relevance - \
                    self.diversity_weight * diversity_penalty

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx
