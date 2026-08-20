"""
Stage 3: Vector DB / retrieval.

Real FAISS (not a mock) -- runs fully in-process, no server, no network.
Uses HNSW (Hierarchical Navigable Small World graph) for approximate
nearest-neighbor search, which is what makes sub-millisecond-to-low-single-
digit-ms retrieval possible even as the corpus grows into the hundreds of
thousands of chunks.

INTERFACE CONTRACT:
    build(chunks, vectors)          -> None
    search(query_vector, k)         -> list[SearchResult]  (+ latency_ms)
"""

import time
from dataclasses import dataclass
import numpy as np
import faiss


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    source_passage_id: str
    score: float          # cosine similarity, higher = more relevant
    metadata: dict


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        # HNSW with 32 neighbors per node -- standard default, good
        # recall/speed tradeoff for corpora up to a few million vectors.
        # IMPORTANT: default FAISS metric is L2 distance (lower = closer),
        # but embedding.py L2-normalizes vectors so that cosine similarity
        # is meaningful -- we explicitly request METRIC_INNER_PRODUCT so
        # that on normalized vectors, "score" means cosine similarity and
        # higher score = more similar, matching guardrails.py's assumption
        # (max(scores) = best match). Mixing these up silently inverts the
        # guardrail threshold logic, so this is called out explicitly.
        self._index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        # efConstruction: graph connectivity built at INDEX TIME (default 40).
        # Raised to 200 after finding, empirically, that FAISS's default
        # leaves the HNSW graph too sparse to stay reliably navigable when
        # the corpus has many near-duplicate/templated passages (common in
        # real corpora too -- syndicated content, boilerplate product
        # descriptions, repeated phrasing) -- at default=40, a query whose
        # true best match sat in a duplicate-heavy cluster was missed
        # entirely (not even in top-100) despite scoring highest by exact
        # cosine similarity; brute-force search confirmed it as the correct
        # top-1 match. Raising efConstruction fixed it. This only costs
        # extra time at BUILD time (one-time, offline), not per-query.
        self._index.hnsw.efConstruction = 200
        self._index.hnsw.efSearch = 64  # search-time breadth; higher = more accurate, slower
        self._chunks = []  # positional list, index i corresponds to vector row i

    def build(self, chunks, vectors: np.ndarray):
        assert vectors.shape[0] == len(chunks)
        assert vectors.shape[1] == self.dim
        self._chunks = list(chunks)
        self._index.add(vectors.astype("float32"))

    def search(self, query_vector: np.ndarray, k: int = 5) -> dict:
        start = time.perf_counter()
        query_vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        scores, indices = self._index.search(query_vector, k)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            c = self._chunks[idx]
            results.append(SearchResult(
                chunk_id=c.chunk_id,
                text=c.text,
                source_passage_id=c.source_passage_id,
                score=float(score),
                metadata=c.metadata,
            ))
        return {"results": results, "latency_ms": elapsed_ms}

    def search_with_filter(self, query_vector: np.ndarray, k: int = 5,
                            topic: str | None = None, over_fetch: int = 50) -> dict:
        """
        Demonstrates metadata-aware retrieval: over-fetch then filter by
        metadata attached during chunking. For large corpora you'd want a
        vector DB with native metadata filtering (Qdrant/Weaviate) instead
        of over-fetch-and-filter, but the principle is the same.

        `over_fetch` matters more than it looks: with a small k (e.g. k=1)
        and a small multiplier, the correct-topic result can sit just
        outside the raw candidate window and get silently missed -- found
        via testing (a k=1 query with a k*4=4 window missed a result that
        WAS in the corpus with the right topic, simply because it ranked
        5th on raw similarity). Defaulting to a fixed, generous floor
        rather than scaling purely off k avoids that failure mode.
        """
        raw = self.search(query_vector, k=max(k * 4, over_fetch))
        if topic is not None:
            filtered = [r for r in raw["results"] if r.metadata.get("topic") == topic]
        else:
            filtered = raw["results"]
        return {"results": filtered[:k], "latency_ms": raw["latency_ms"]}
