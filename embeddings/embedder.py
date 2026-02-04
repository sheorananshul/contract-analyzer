# embeddings/embedder.py
from typing import List
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(
    texts: List[str],
    model: str = "text-embedding-3-small"
) -> List[List[float]]:
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    return [item.embedding for item in response.data]
