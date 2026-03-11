"""Full safety pipeline composing all trust and safety checks."""

import logging
from typing import List, Dict, Any, Tuple

from trust_safety.toxicity_classifier import ToxicityClassifier
from trust_safety.spam_detector import SpamDetector
from trust_safety.policy_filter import PolicyFilter

logger = logging.getLogger(__name__)


class SafetyPipeline:
    """
    Runs the full safety stack on candidate videos before serving.
    Applies: spam detection ? toxicity classification ? policy filtering.
    """

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or {}
        self._spam = SpamDetector(spam_threshold=cfg.get("spam_threshold", 0.8))
        self._toxicity = ToxicityClassifier(threshold=cfg.get("toxicity_threshold", 0.7))
        self._policy = PolicyFilter(
            blocked_categories=cfg.get("blocked_categories", []),
            min_creator_trust_score=cfg.get("min_creator_trust_score", 0.5),
        )
        self._blocked_count = 0
        self._total_processed = 0

    async def run(self, candidates: List[Dict[str, Any]],
                  video_metadata: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run full safety pipeline and return safe candidates."""
        self._total_processed += len(candidates)
        safe = []
        for cand in candidates:
            vid = cand.get("video_id", "")
            meta = video_metadata.get(vid, {})
            is_safe, reason = await self._check_safety(cand, meta)
            if is_safe:
                safe.append(cand)
            else:
                cand["blocked_reason"] = reason
                self._blocked_count += 1
                logger.debug(f"Blocked video {vid}: {reason}")

        safe = self._policy.filter_candidates(safe, video_metadata)
        return safe

    async def _check_safety(self, candidate: Dict[str, Any],
                             meta: Dict[str, Any]) -> Tuple[bool, str]:
        title = meta.get("title", "")
        description = meta.get("description", "")

        # Spam check
        is_spam, spam_score = self._spam.is_spam_content(title, description)
        if is_spam:
            return False, f"spam:{spam_score:.2f}"

        # Toxicity check on title
        is_toxic, tox_reason = self._toxicity.is_toxic(title)
        if is_toxic:
            return False, f"toxicity:{tox_reason}"

        return True, ""

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_processed": self._total_processed,
            "blocked_count": self._blocked_count,
            "block_rate": self._blocked_count / max(self._total_processed, 1),
        }
