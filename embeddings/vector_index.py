import faiss
import numpy as np


class VectorIndex:

    def __init__(self, dim=32):

        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)

        # simulate video embeddings
        self.video_ids = []
        vectors = []

        for i in range(1000):
            self.video_ids.append(i)
            vectors.append(np.random.rand(dim))

        vectors = np.array(vectors).astype("float32")

        self.index.add(vectors)

    def search(self, user_embedding, k=50):

        user_embedding = np.array([user_embedding]).astype("float32")

        distances, indices = self.index.search(user_embedding, k)

        results = []

        for idx in indices[0]:
            results.append({"video_id": int(self.video_ids[idx])})

        return results