from __future__ import annotations

import math
import re
from typing import Any

from .embeddings import EmbeddingsUnavailable, LocalEmbedder


def terms(text: str) -> list[str]:
    def normalize(word: str) -> str:
        word = word.lower()
        if word.startswith(("dedup", "deduplicat", "duplicate", "duplicat")):
            return "duplicate"
        if len(word) > 5 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 5 and word.endswith("ing"):
            root = word[:-3]
            if len(root) > 2 and root[-1] == root[-2]:
                root = root[:-1]
            return root
        if len(word) > 4 and word.endswith("ied"):
            return word[:-3] + "y"
        if len(word) > 4 and word.endswith("ed"):
            root = word[:-2]
            return root + "e" if root.endswith("at") else root
        if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

    return [normalize(word) for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", text) if len(word) > 2]


def semantic_scores(question: str, chunks: list[dict[str, Any]], vectors: dict[str, list[float]],
                    embedder: LocalEmbedder) -> dict[str, float] | None:
    if not vectors:
        return None
    query = embedder.embed_query(question)
    query_norm = math.sqrt(sum(value * value for value in query))
    if not query_norm:
        return None
    scores: dict[str, float] = {}
    for chunk in chunks:
        vector = vectors.get(chunk.get("id", ""))
        if not vector or len(vector) != len(query):
            continue
        vector_norm = math.sqrt(sum(value * value for value in vector))
        if vector_norm:
            scores[chunk["id"]] = sum(a * b for a, b in zip(query, vector)) / (query_norm * vector_norm)
    return scores or None


def hybrid_rank(lexical: list[tuple[float, dict[str, Any]]],
                semantic: dict[str, float] | None,
                candidates: list[dict[str, Any]], limit: int,
                semantic_weight: float = 0.7) -> list[dict[str, Any]]:
    if not semantic:
        return [{**chunk, "score": round(score, 2)} for score, chunk in lexical[:limit]]

    lexical_ranks = {chunk.get("id"): rank for rank, (_, chunk) in enumerate(lexical, 1) if chunk.get("id")}
    semantic_order = sorted(semantic, key=semantic.get, reverse=True)
    semantic_ranks = {chunk_id: rank for rank, chunk_id in enumerate(semantic_order, 1)}
    by_id = {chunk.get("id"): chunk for chunk in candidates if chunk.get("id")}
    for _, chunk in lexical:
        if chunk.get("id"):
            by_id[chunk["id"]] = chunk
    combined = []
    lexical_count = max(1, len(lexical_ranks))
    semantic_count = max(1, len(semantic_ranks))
    for chunk_id in set(lexical_ranks) | set(semantic_ranks):
        chunk = by_id.get(chunk_id)
        if not chunk:
            continue
        lex_rank = lexical_ranks.get(chunk_id)
        sem_rank = semantic_ranks.get(chunk_id)
        lexical_score = (lexical_count - lex_rank + 1) / lexical_count if lex_rank else 0
        semantic_score = (semantic_count - sem_rank + 1) / semantic_count if sem_rank else 0
        score = semantic_weight * semantic_score + (1 - semantic_weight) * lexical_score
        combined.append((score, chunk))
    combined.sort(key=lambda item: item[0], reverse=True)
    return [{**chunk, "score": round(score, 5)} for score, chunk in combined[:limit]]
