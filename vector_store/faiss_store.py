from typing import List, Dict, Tuple
import numpy as np
import faiss


class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine-like if vectors normalized
        self.text_by_id: Dict[int, str] = {}
        self.meta_by_id: Dict[int, dict] = {}

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def add(self, embeddings: List[List[float]], texts: List[str], metas: List[dict] = None):
        vecs = np.array(embeddings, dtype="float32")
        vecs = self._normalize(vecs)

        start_id = len(self.text_by_id)
        ids = np.arange(start_id, start_id + len(texts)).astype("int64")

        self.index.add(vecs)
        for i, t in enumerate(texts):
            self.text_by_id[int(ids[i])] = t
            self.meta_by_id[int(ids[i])] = metas[i] if metas else {}

    def search(self, query_embedding: List[float], top_k: int = 6) -> List[Tuple[int, float, str, dict]]:
        q = np.array([query_embedding], dtype="float32")
        q = self._normalize(q)

        scores, idxs = self.index.search(q, top_k)
        results = []
        for doc_id, score in zip(idxs[0], scores[0]):
            if doc_id == -1:
                continue
            results.append((int(doc_id), float(score), self.text_by_id[int(doc_id)], self.meta_by_id[int(doc_id)]))
        return results
