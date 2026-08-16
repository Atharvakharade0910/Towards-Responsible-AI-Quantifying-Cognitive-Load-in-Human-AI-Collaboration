"""Dependency-light hybrid RAG pipeline for the assessment knowledge base."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any


EMBEDDING_DIMENSION = 384
LEXICAL_WEIGHT = 0.45
SEMANTIC_WEIGHT = 0.55
BM25_K1 = 1.5
BM25_B = 0.75
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def _tokens(text: str) -> list[str]:
    words = [word.lower() for word in WORD_RE.findall(text or "")]
    # Character n-grams add a little robustness for related word forms without
    # adding a heavyweight ML dependency to the assessment server.
    grams: list[str] = []
    for word in words:
        grams.extend(f"{word[i:i + 3]}" for i in range(max(0, len(word) - 2)))
    return words + grams


def _lexical_tokens(text: str) -> list[str]:
    """Return word tokens for exact lexical/BM25 matching."""
    return [word.lower() for word in WORD_RE.findall(text or "")]


def embed_text(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSION
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSION
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 8) for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    return max(-1.0, min(1.0, sum(left[i] * right[i] for i in range(size))))


def _bm25_score(query_tokens: list[str], document_tokens: list[str], document_frequency: dict[str, int], document_count: int, average_length: float) -> float:
    if not query_tokens or not document_tokens or not document_count:
        return 0.0
    frequencies: dict[str, int] = {}
    for token in document_tokens:
        frequencies[token] = frequencies.get(token, 0) + 1
    length_ratio = len(document_tokens) / (average_length or 1.0)
    score = 0.0
    for token in set(query_tokens):
        term_frequency = frequencies.get(token, 0)
        if not term_frequency:
            continue
        df = document_frequency.get(token, 0)
        idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
        score += idf * (
            (term_frequency * (BM25_K1 + 1.0))
            / (term_frequency + BM25_K1 * (1.0 - BM25_B + BM25_B * length_ratio))
        )
    return score


def chunk_text(text: str, words_per_chunk: int = 260, overlap: int = 45) -> list[str]:
    words = (text or "").split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + words_per_chunk)
        chunks.append(" ".join(words[start:end]).strip())
        if end >= len(words):
            break
        start = max(start + 1, end - overlap)
    return chunks


def retrieve(query: str, rows: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    query_embedding = embed_text(query)
    query_tokens = _lexical_tokens(query)
    tokenized_rows = [_lexical_tokens(str(row.get("chunk_text") or "")) for row in rows]
    document_frequency: dict[str, int] = {}
    for tokens in tokenized_rows:
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    average_length = sum(len(tokens) for tokens in tokenized_rows) / len(tokenized_rows) if tokenized_rows else 0.0
    lexical_scores = [
        _bm25_score(query_tokens, tokens, document_frequency, len(rows), average_length)
        for tokens in tokenized_rows
    ]
    max_lexical_score = max(lexical_scores, default=0.0)
    ranked: list[dict[str, Any]] = []
    for row, lexical_score in zip(rows, lexical_scores):
        try:
            stored = row.get("embedding") or "[]"
            import json
            embedding = json.loads(stored) if isinstance(stored, str) else stored
            semantic_score = max(0.0, cosine_similarity(query_embedding, [float(value) for value in embedding]))
        except (TypeError, ValueError, json.JSONDecodeError):
            semantic_score = 0.0
        lexical_similarity = lexical_score / max_lexical_score if max_lexical_score else 0.0
        hybrid_score = (LEXICAL_WEIGHT * lexical_similarity) + (SEMANTIC_WEIGHT * semantic_score)
        ranked.append({
            **row,
            "similarity": round(hybrid_score, 4),
            "lexical_similarity": round(lexical_similarity, 4),
            "semantic_similarity": round(semantic_score, 4),
        })
    ranked.sort(key=lambda row: float(row.get("similarity") or 0.0), reverse=True)
    return ranked[:max(1, limit)]


def classify_direction(similarity: float, faithfulness: float) -> tuple[str, str]:
    score = round((max(0.0, similarity) * 0.55) + (max(0.0, faithfulness) * 0.45), 4)
    if score >= 0.62:
        return str(score), "correct"
    if score >= 0.38:
        return str(score), "partial"
    return str(score), "off_track"
