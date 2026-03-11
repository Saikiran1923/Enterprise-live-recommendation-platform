"""Generates synthetic user, video, and interaction data for development."""

import json
import random
import time
import uuid
import argparse
import os
from typing import List, Dict, Any

CATEGORIES = ["gaming", "music", "sports", "news", "education",
              "comedy", "lifestyle", "tech", "cooking", "travel"]
PLATFORMS = ["web", "ios", "android", "tv"]


def generate_users(n: int) -> List[Dict[str, Any]]:
    users = []
    for i in range(n):
        users.append({
            "user_id": f"user_{i:06d}",
            "created_at": time.time() - random.randint(0, 365 * 86400),
            "country": random.choice(["US", "GB", "DE", "JP", "BR", "IN"]),
            "language": random.choice(["en", "de", "ja", "pt", "hi"]),
            "is_creator": random.random() < 0.05,
            "subscription_tier": random.randint(0, 2),
        })
    return users


def generate_videos(n: int, users: List[Dict]) -> List[Dict[str, Any]]:
    videos = []
    creator_ids = [u["user_id"] for u in users if u.get("is_creator")]
    for i in range(n):
        upload_time = time.time() - random.randint(0, 30 * 86400)
        videos.append({
            "video_id": f"video_{i:08d}",
            "creator_id": random.choice(creator_ids) if creator_ids else f"creator_{i}",
            "title": f"Sample Video {i} - {random.choice(CATEGORIES).title()} Content",
            "category": random.choice(CATEGORIES),
            "duration_sec": random.choice([30, 60, 180, 300, 600, 1800, 3600]),
            "upload_timestamp": upload_time,
            "language": random.choice(["en", "de", "ja", "pt"]),
            "tags": random.sample(CATEGORIES, k=random.randint(1, 4)),
            "is_active": True,
            "creator_trust_score": round(random.uniform(0.5, 1.0), 2),
            "creator_follower_count": random.randint(100, 5000000),
        })
    return videos


def generate_interactions(users: List[Dict], videos: List[Dict],
                           n: int) -> List[Dict[str, Any]]:
    events = []
    user_prefs = {u["user_id"]: random.sample(CATEGORIES, k=3) for u in users}

    for _ in range(n):
        user = random.choice(users)
        uid = user["user_id"]
        prefs = user_prefs[uid]
        # Weighted video selection: prefer user's category preferences
        video = random.choice([v for v in videos if v["category"] in prefs] or videos)

        duration = video["duration_sec"]
        watch_rate = max(0.05, min(1.0, random.betavariate(2, 2)))
        watch_sec = duration * watch_rate

        events.append({
            "event_id": str(uuid.uuid4()),
            "user_id": uid,
            "video_id": video["video_id"],
            "event_type": "video_view",
            "watch_duration_sec": round(watch_sec, 1),
            "video_duration_sec": duration,
            "completion_rate": round(watch_rate, 3),
            "category": video["category"],
            "creator_id": video["creator_id"],
            "session_id": f"session_{uid}_{int(time.time())}",
            "platform": random.choice(PLATFORMS),
            "timestamp": time.time() - random.randint(0, 30 * 86400),
            "liked": random.random() < 0.08,
            "shared": random.random() < 0.02,
        })
    return events


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic platform data")
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--videos", type=int, default=5000)
    parser.add_argument("--interactions", type=int, default=50000)
    parser.add_argument("--output-dir", type=str, default="data/raw")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.users} users...")
    users = generate_users(args.users)
    with open(f"{args.output_dir}/users.json", "w") as f:
        json.dump(users, f, indent=2)

    print(f"Generating {args.videos} videos...")
    videos = generate_videos(args.videos, users)
    with open(f"{args.output_dir}/videos.json", "w") as f:
        json.dump(videos, f, indent=2)

    print(f"Generating {args.interactions} interactions...")
    interactions = generate_interactions(users, videos, args.interactions)
    with open(f"{args.output_dir}/interactions.json", "w") as f:
        json.dump(interactions, f, indent=2)

    print(f"Done! Data written to {args.output_dir}/")


if __name__ == "__main__":
    main()
