from typing import List, Dict
from embeddings.embedder import embed_texts
from vector_store.faiss_store import FaissVectorStore


def retrieve_clauses(
    store: FaissVectorStore,
    requirement_name: str,
    requirement_description: str,
    controls: List[str],
    top_k: int = 12,
    min_score: float = 0.25,   # <-- add threshold (tune 0.25–0.40)
) -> List[Dict]:
    query = (
        f"Requirement: {requirement_name}\n"
        f"Description: {requirement_description}\n"
        f"Controls: {', '.join(controls)}"
    )

    q_emb = embed_texts([query])[0]
    hits = store.search(q_emb, top_k=top_k)

    results = []
    for doc_id, score, text, meta in hits:
        results.append({
            "chunk_id": meta.get("chunk_id", doc_id),
            "label": meta.get("label", f"Chunk {doc_id}"),  
            "score": score,
            "text": text,
        })


    # ✅ Filter by relevance
    results = [r for r in results if r["score"] >= min_score]

    return results
