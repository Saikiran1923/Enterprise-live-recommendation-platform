"Lightweight candidate scoring before ranking."

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class PreRankingService:
    
    def __init__(self, max_candidates: int = 200):
        self.max_candidates = max_candidates

    async def prerank(
        self,
        candidates: List[Dict[str, Any]],
        user_features: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:

        # Example lightweight scoring
        for c in candidates:

            score = c.get("retrieval_score", 0)

            # freshness boost
            if context.get("is_live_session"):
                score *= 1.1

            c["prerank_score"] = score

        candidates.sort(
            key=lambda x: x["prerank_score"],
            reverse=True
        )

        return candidates[:self.max_candidates]