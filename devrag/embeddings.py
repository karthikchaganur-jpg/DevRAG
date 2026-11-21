from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from .config import EMBEDDING_MODEL, EMBEDDING_MODEL_CACHE


class EmbeddingsUnavailable(RuntimeError):
    pass


class LocalEmbedder:
    """CPU embeddings through FastEmbed; no hosted API key is used."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, cache_dir=EMBEDDING_MODEL_CACHE):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise EmbeddingsUnavailable("Install fastembed to enable local semantic search.") from exc
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=str(self.cache_dir))
        except Exception as exc:
            raise EmbeddingsUnavailable(f"Could not load the local embedding model: {exc}") from exc
        return self._model

    def embed_passages(self, texts: Iterable[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        try:
            return [vector.tolist() if hasattr(vector, "tolist") else list(vector) for vector in self._load().passage_embed(values)]
        except EmbeddingsUnavailable:
            raise
        except Exception as exc:
            raise EmbeddingsUnavailable(f"Could not generate local embeddings: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            vector = next(iter(self._load().query_embed(text)))
            return vector.tolist() if hasattr(vector, "tolist") else list(vector)
        except EmbeddingsUnavailable:
            raise
        except Exception as exc:
            raise EmbeddingsUnavailable(f"Could not embed the question locally: {exc}") from exc


@lru_cache(maxsize=1)
def get_embedder() -> LocalEmbedder:
    return LocalEmbedder()
