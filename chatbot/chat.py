import os
from typing import List, Dict, Tuple
from openai import OpenAI
from embeddings.embedder import embed_texts

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def retrieve_for_chat(store, question: str, top_k: int = 6) -> List[Dict]:
    """
    Retrieve top_k relevant chunks from FAISS for a chat question.
    Expects store.search(query_embedding, top_k) -> list of (doc_id, score, text, meta)
    """
    q_emb = embed_texts([question], model="text-embedding-3-small")[0]
    hits = store.search(q_emb, top_k=top_k)

    retrieved = []
    for doc_id, score, text, meta in hits:
        retrieved.append({
            "chunk_id": meta.get("chunk_id", doc_id),
            "score": score,
            "text": text
        })
    return retrieved


def answer_question_with_rag(
    store,
    question: str,
    top_k: int = 6,
    model: str = "gpt-4o-mini",
) -> Tuple[str, List[Dict]]:
    """
    Returns (answer, retrieved_chunks).
    Answer is grounded only in retrieved contract text.
    """
    retrieved = retrieve_for_chat(store, question, top_k=top_k)

    context = "\n\n".join(
        [f"[Chunk {c['chunk_id']}]\n{c['text']}" for c in retrieved]
    )

    prompt = f"""
You are a contract assistant. Answer the user’s question using ONLY the contract text provided.

Rules:
- If the answer is not in the text, say: "I can't find that in the provided contract."
- Quote short relevant phrases where helpful.
- When you use information from the contract, cite chunk ids like (Chunk 3) inline.
- Do not invent facts.

Contract Text:
{context}

User Question:
{question}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You answer strictly from provided contract text and cite chunk ids."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    answer = resp.choices[0].message.content.strip()
    return answer, retrieved
