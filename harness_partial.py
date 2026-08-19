"""
Harness for stages 1-6 (chunking through post-generation guardrail).
STT (stage 7) is not wired in yet -- this harness takes TEXT questions
directly, exactly as if STT had already run and hitting this with real
audio later just means adding one more stage in front, no restructuring.

This module also doubles as the "structured orchestration" requirement:
each stage's output is a typed/dict object with an explicit shape, errors
are caught per-stage, and every stage's latency is logged individually.
"""

import functools
import time
from collections import defaultdict

from chunking import chunk_sentence_aware
from embedding import TfidfEmbedder
from vector_store import VectorStore
from guardrails import pre_generation_check, post_generation_check_selfreport
from generation import MockLLM, call_with_retry


class LatencyLog:
    def __init__(self):
        self._data = defaultdict(list)

    def record(self, stage: str, ms: float):
        self._data[stage].append(ms)

    def stages(self):
        return list(self._data.keys())

    def values(self, stage: str):
        return self._data[stage]

    def as_dict(self):
        return dict(self._data)


def timed(stage_name, log: LatencyLog):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.record(stage_name, elapsed_ms)
            return result
        return wrapper
    return decorator


class PartialRAGHarness:
    """
    Wires: chunking (offline, one-time) -> embedding -> vector search ->
    pre-guardrail -> generation (mock) -> post-guardrail.
    """

    def __init__(self, passages, hallucination_rate=0.15, simulate_llm_latency=True):
        self.log = LatencyLog()
        self.llm = MockLLM(simulate_latency=simulate_llm_latency,
                            hallucination_rate=hallucination_rate)

        # --- offline indexing step (not part of per-query latency) ---
        self.chunks = chunk_sentence_aware(passages, max_sentences_per_chunk=2)
        self.embedder = TfidfEmbedder(dim=64)
        self.embedder.fit([c.text for c in self.chunks])
        chunk_vectors = self.embedder.embed_batch([c.text for c in self.chunks])
        self.store = VectorStore(dim=self.embedder.dim)
        self.store.build(self.chunks, chunk_vectors)

    def answer(self, question: str, k: int = 5, min_score: float = 0.2) -> dict:
        """
        Runs one query through stages 2-6 (embedding through post-guardrail),
        with every stage timed individually and an explicit refusal reason
        if any guardrail trips.
        """
        trace = {"question": question}

        embed_fn = timed("embed_query", self.log)(self.embedder.embed)
        query_vec = embed_fn(question)

        search_fn = timed("vector_search", self.log)(
            lambda qv: self.store.search(qv, k=k)
        )
        search_out = search_fn(query_vec)
        results = search_out["results"]
        trace["retrieved"] = [(r.chunk_id, round(r.score, 3)) for r in results]

        pre_check = pre_generation_check(results, min_score=min_score)
        trace["pre_guardrail"] = pre_check.reason
        if not pre_check.passed:
            trace["answer"] = "I don't have relevant information to answer that."
            trace["final_status"] = "refused_" + pre_check.reason
            return trace

        context_texts = [r.text for r in results]
        context_ids = [r.chunk_id for r in results]

        gen_fn = timed("generation", self.log)(
            lambda: call_with_retry(self.llm, question, context_texts, context_ids)
        )
        gen_out = gen_fn()
        trace["raw_answer"] = gen_out["answer"]
        trace["used_chunk_ids"] = gen_out["used_chunk_ids"]

        post_start = time.perf_counter()
        post_check = post_generation_check_selfreport(gen_out)
        self.log.record("post_guardrail", (time.perf_counter() - post_start) * 1000)

        trace["post_guardrail"] = post_check.reason
        if not post_check.passed:
            trace["answer"] = "I found some context, but I'm not confident the answer is fully supported by it."
            trace["final_status"] = "refused_" + post_check.reason
        else:
            trace["answer"] = gen_out["answer"]
            trace["final_status"] = "answered"

        return trace
