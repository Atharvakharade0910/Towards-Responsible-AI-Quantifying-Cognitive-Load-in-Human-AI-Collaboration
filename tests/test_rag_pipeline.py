from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from rag_pipeline import embed_text, retrieve


def test_embedding_is_custom_384_dimension_hash_vector() -> None:
    vector = embed_text("cosine similarity and retrieval")
    assert len(vector) == 384
    assert abs(sum(value * value for value in vector) - 1.0) < 0.001


def test_retrieve_combines_lexical_and_semantic_scores() -> None:
    rows = [
        {"chunk_id": "exact", "chunk_text": "A cognitive load assessment measures mental effort.", "embedding": json.dumps(embed_text("cognitive load assessment"))},
        {"chunk_id": "other", "chunk_text": "A participant completes a visual attention task.", "embedding": json.dumps(embed_text("visual attention task"))},
    ]
    results = retrieve("cognitive load assessment", rows, limit=2)
    assert results[0]["chunk_id"] == "exact"
    assert results[0]["lexical_similarity"] > 0
    assert results[0]["semantic_similarity"] > 0
    assert results[0]["similarity"] > 0
