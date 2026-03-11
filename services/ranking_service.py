import random

class RankingService:
    def __init__(self):
        pass

    def rank(self, user_features, candidates):
        ranked = []
        for video in candidates:
            score = random.random()
            ranked.append({
                "video_id": video["video_id"],
                "ranking_score": round(score, 3),
                "retrieval_score": video.get("retrieval_score"),
                "source": video.get("source"),
            })
        ranked.sort(key=lambda x: x["ranking_score"], reverse=True)
        return ranked