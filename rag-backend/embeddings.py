
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

_embedder = None


def get_embedder():
    global _embedder

    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)

    return _embedder
