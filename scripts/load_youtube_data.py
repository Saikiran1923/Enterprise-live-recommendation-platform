import pandas as pd
import asyncio
from storage.feature_cache_store import FeatureCacheStore

store = FeatureCacheStore()

async def load():

    df = pd.read_csv("data/raw/USvideos.csv")

    for _, row in df.head(1000).iterrows():

        features = {
            "video_id": row["video_id"],
            "title": row["title"],
            "views": int(row["views"]),
            "likes": int(row["likes"])
        }

        await store.set("video", row["video_id"], features)

    print("videos loaded")

asyncio.run(load())