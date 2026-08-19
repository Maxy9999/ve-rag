"""
Stage 8: Full harness -- STT through post-generation guardrail, all 7
pipeline stages wired together with:
  - typed/structured per-stage inputs & outputs (no raw string-passing)
  - per-stage timing (feeds stage 9 directly)
  - per-stage error handling with an explicit, loggable failure reason
  - retries where a stage involves a network call (STT, generation)
  - one QueryTrace object per query, capturing the FULL decision path --
    this is what makes the guardrail behavior auditable/demoable rather
    than just "trust me, it works."

Swap MockSTT -> SarvamSTT/ElevenLabsSTT and MockLLM -> a real API-backed
generator on your machine; every other line in this file stays the same,
because stage 2-7 all talk to each other through the same fixed dict/
dataclass shapes established in embedding.py, vector_store.py,
guardrails.py, generation.py, and stt.py.
"""

import functools
import time
from collections import defaultdict
from dataclasses import dataclass, field

from chunking import chunk_sentence_aware
from embedding import TfidfEmbedder
from vector_store import VectorStore
from guardrails import stt_confidence_check, pre_generation_check, post_generation_check_selfreport
from generation import MockLLM, call_with_retry
from stt import MockSTT


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


@dataclass
class QueryTrace:
    """
    Full audit trail for one query through all 7 stages. This is the
    structured-output requirement made concrete: every downstream consumer
    (a UI, a log aggregator, this project's own test suite) reads from
    this object instead of parsing free text.
    """
    input_text: str = ""          # what was "said" (sandbox: given directly, or via MockSTT)
    transcript: str = ""          # STT output
    stt_confidence: float | None = None
    retrieved: list = field(default_factory=list)   # [(chunk_id, score), ...]
    used_chunk_ids: list = field(default_factory=list)
    raw_answer: str = ""
    final_answer: str = ""
    final_status: str = ""        # "answered" | "refused_<reason>"
    refusal_reason: str = ""
    stage_latencies_ms: dict = field(default_factory=dict)
    total_latency_ms: float = 0.0


class RAGHarness:
    """
    Full 7-stage harness: STT -> embed -> vector search -> pre-guardrail
    -> generation -> post-guardrail -> final answer.
    """

    def __init__(self, passages, stt=None, llm=None,
                 hallucination_rate=0.15, simulate_latency=True,
                 chunk_strategy=chunk_sentence_aware):
        self.log = LatencyLog()
        self.stt = stt or MockSTT(simulate_latency=simulate_latency)
        self.llm = llm or MockLLM(simulate_latency=simulate_latency,
                                   hallucination_rate=hallucination_rate)

        # --- offline indexing (not counted in per-query latency) ---
        self.chunks = chunk_strategy(passages)
        self.embedder = TfidfEmbedder(dim=64)
        self.embedder.fit([c.text for c in self.chunks])
        chunk_vectors = self.embedder.embed_batch([c.text for c in self.chunks])
        self.store = VectorStore(dim=self.embedder.dim)
        self.store.build(self.chunks, chunk_vectors)

    def _refuse(self, trace: QueryTrace, reason: str, message: str, t_start: float) -> QueryTrace:
        trace.final_status = "refused_" + reason
        trace.refusal_reason = reason
        trace.final_answer = message
        trace.stage_latencies_ms = {k: v[-1] if v else None for k, v in self.log.as_dict().items()}
        trace.total_latency_ms = (time.perf_counter() - t_start) * 1000
        return trace

    def answer(self, spoken_text: str, k: int = 5,
               min_stt_confidence: float = 0.6, min_retrieval_score: float = 0.15) -> QueryTrace:
        """
        `spoken_text` stands in for "the ground-truth text of what the user
        said" -- MockSTT.transcribe_text() simulates STT latency/confidence
        around it. On your machine with real audio: call
        SarvamSTT.transcribe(audio_path) instead, same downstream code.
        """
        t_start = time.perf_counter()
        trace = QueryTrace(input_text=spoken_text)

        # --- Stage 7: STT ---
        stt_fn = timed("stt", self.log)(self.stt.transcribe_text)
        stt_out = stt_fn(spoken_text)
        trace.transcript = stt_out["text"]
        trace.stt_confidence = stt_out["confidence"]

        stt_check = stt_confidence_check(stt_out, min_confidence=min_stt_confidence)
        if not stt_check.passed:
            return self._refuse(trace, stt_check.reason,
                                 "I didn't catch that clearly -- could you repeat the question?",
                                 t_start)

        # --- Stage 2: embed ---
        embed_fn = timed("embed_query", self.log)(self.embedder.embed)
        query_vec = embed_fn(trace.transcript)

        # --- Stage 3: vector search ---
        search_fn = timed("vector_search", self.log)(lambda qv: self.store.search(qv, k=k))
        search_out = search_fn(query_vec)
        results = search_out["results"]
        trace.retrieved = [(r.chunk_id, round(r.score, 3)) for r in results]

        # --- Stage 4: pre-generation guardrail ---
        pre_check = pre_generation_check(results, min_score=min_retrieval_score)
        if not pre_check.passed:
            return self._refuse(trace, pre_check.reason,
                                 "I don't have relevant information to answer that.",
                                 t_start)

        context_texts = [r.text for r in results]
        context_ids = [r.chunk_id for r in results]

        # --- Stage 5: generation ---
        gen_fn = timed("generation", self.log)(
            lambda: call_with_retry(self.llm, trace.transcript, context_texts, context_ids)
        )
        gen_out = gen_fn()
        trace.raw_answer = gen_out.get("answer") or ""
        trace.used_chunk_ids = gen_out.get("used_chunk_ids", [])

        if gen_out.get("error"):
            return self._refuse(trace, "generation_error",
                                 "Something went wrong generating an answer -- please try again.",
                                 t_start)

        # --- Stage 6: post-generation guardrail ---
        post_start = time.perf_counter()
        post_check = post_generation_check_selfreport(gen_out)
        self.log.record("post_guardrail", (time.perf_counter() - post_start) * 1000)

        if not post_check.passed:
            return self._refuse(trace, post_check.reason,
                                 "I found related context, but I'm not confident the answer is "
                                 "fully supported by it -- I'd rather not guess.",
                                 t_start)

        trace.final_status = "answered"
        trace.final_answer = trace.raw_answer
        trace.stage_latencies_ms = {k: v[-1] if v else None for k, v in self.log.as_dict().items()}
        trace.total_latency_ms = (time.perf_counter() - t_start) * 1000
        return trace
