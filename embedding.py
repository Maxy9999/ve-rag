"""
Stage 2: Embedding.

INTERFACE CONTRACT (this is the important part -- keep this shape stable):
    fit(texts: list[str]) -> None            # build vocabulary/model state
    embed(text: str) -> np.ndarray           # 1 vector, fixed dimension
    embed_batch(texts: list[str]) -> np.ndarray  # N vectors, same dimension
    dim -> int                                # vector dimensionality

This sandbox cannot reach huggingface.co, so `TfidfEmbedder` below is a
LOCAL, dependency-free stand-in (scikit-learn ships its own TF-IDF logic,
no model download needed) that proves the pipeline WIRING is correct.

On your machine, swap it for `SentenceTransformerEmbedder` (implemented
below too, commented) -- same interface, so nothing else in the pipeline
(vector_store.py, harness.py) needs to change. That's the whole point of
fixing the interface first.
"""

import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class TfidfEmbedder:
    """
    Local stand-in for a real sentence-embedding model. TF-IDF + SVD
    (a poor-man's LSA) gives fixed-length dense vectors purely from local
    text statistics -- no network call, no model download -- so retrieval
    logic can be built and tested fully offline. It will NOT capture deep
    semantic similarity the way a real transformer embedding does (e.g.
    "car" vs "automobile" won't be close), which is an explicit, documented
    limitation -- not something to be surprised by later.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        # stop_words='english' matters more than it looks: without it,
        # off-topic queries score misleadingly high against ANY passage
        # purely because they share function words ("what", "is", "does")
        # -- there's no clean score gap to threshold on for guardrailing.
        # With stopwords stripped, TF-IDF only responds to real shared
        # vocabulary, so off-topic queries correctly collapse to ~0
        # similarity. Verified empirically -- see harness guardrail demo.
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._svd = TruncatedSVD(n_components=dim, random_state=0)
        self._fitted = False

    def fit(self, texts: list[str]):
        tfidf = self._vectorizer.fit_transform(texts)
        # SVD components can't exceed min(n_samples, n_features) - 1
        n_components = min(self.dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        n_components = max(n_components, 1)
        if n_components != self.dim:
            self._svd = TruncatedSVD(n_components=n_components, random_state=0)
            self.dim = n_components
        self._svd.fit(tfidf)
        self._fitted = True

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        assert self._fitted, "call .fit(all_chunk_texts) once before embedding"
        tfidf = self._vectorizer.transform(texts)
        vecs = self._svd.transform(tfidf).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms  # L2-normalize so cosine sim == dot product

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]


class SentenceTransformerEmbedder:
    """
    REAL embedder -- use this on your machine where huggingface.co is
    reachable. Not runnable in this sandbox (model download blocked).
    Same interface as TfidfEmbedder above, so it's a drop-in swap.

    Install: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer  # local import: optional dep
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def fit(self, texts: list[str]):
        pass  # pretrained model, no fitting step needed

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True).astype("float32")

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]


def timed_embed_query(embedder, text: str) -> dict:
    start = time.perf_counter()
    vec = embedder.embed(text)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {"vector": vec, "latency_ms": elapsed_ms}
