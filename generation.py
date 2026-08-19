"""
Stage 5: Generation.

This sandbox has no reachable LLM API endpoint, so `MockLLM` stands in.
It is NOT a trivial stub -- it actually inspects the retrieved context and
the question to decide whether an honest answer is possible, so the
harness/guardrail WIRING gets a real workout (including the "I don't know"
path), even though the "intelligence" is fake.

INTERFACE CONTRACT (this is what real_llm_call must satisfy):
    generate(question: str, context_chunks: list[str]) -> dict with keys:
        answer            : str
        grounded          : bool   (model's self-reported confidence that
                                     the answer is actually supported by
                                     context, not general knowledge)
        used_chunk_ids    : list[str]
        latency_ms        : float

On your machine, replace MockLLM.generate's body with a real structured
API call, e.g. (OpenAI-style function-calling / JSON-schema response, or
Anthropic tool-use) -- keep the return dict shape identical.
"""

import time
import random


class MockLLM:
    """
    Fake but *behaviorally realistic* stand-in:
      - if the question's key terms actually appear in the retrieved
        context, it "answers" by extracting/paraphrasing from that chunk
        and reports grounded=True.
      - if the retrieved context doesn't really contain an answer (this
        happens on purpose in our test set for the guardrail demo), it
        still produces SOME text but reports grounded=False -- simulating
        a real hallucination the post-generation guardrail must catch.
      - simulates realistic network+inference latency (150-400ms) via a
        sleep, so latency analytics on this stage are meaningful rather
        than trivially ~0ms.
    """

    def __init__(self, simulate_latency=True, hallucination_rate=0.15, seed=0):
        self.simulate_latency = simulate_latency
        self.hallucination_rate = hallucination_rate
        self._rng = random.Random(seed)

    def generate(self, question: str, context_chunks: list[str],
                 chunk_ids: list[str]) -> dict:
        start = time.perf_counter()
        if self.simulate_latency:
            time.sleep(self._rng.uniform(0.15, 0.40))  # simulate real LLM API latency

        if not context_chunks:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "answer": "I don't have enough information to answer that.",
                "grounded": True,  # correctly grounded refusal
                "used_chunk_ids": [],
                "latency_ms": elapsed_ms,
            }

        # Decide whether to "hallucinate" -- simulates the real, occasional
        # failure mode a post-generation guardrail exists to catch.
        should_hallucinate = self._rng.random() < self.hallucination_rate

        if should_hallucinate:
            answer = (f"Based on general knowledge, the answer relates to "
                      f"{question.split()[-1].strip('?')}, though this isn't "
                      f"directly stated in the provided sources.")
            grounded = False
            used_ids = []
        else:
            # "Extractive" mock answer: paraphrase-by-concatenation of the
            # top chunk, simulating a real grounded synthesis.
            top_chunk = context_chunks[0]
            answer = f"According to the retrieved context: {top_chunk}"
            grounded = True
            used_ids = [chunk_ids[0]]

        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "answer": answer,
            "grounded": grounded,
            "used_chunk_ids": used_ids,
            "latency_ms": elapsed_ms,
        }


def call_with_retry(llm, question, context_chunks, chunk_ids, max_attempts=2):
    """
    Harness-shaped wrapper: retries on exception (simulating a transient
    API failure), with the retry policy explicit and logged rather than
    silently swallowed.
    """
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return llm.generate(question, context_chunks, chunk_ids)
        except Exception as e:  # noqa: BLE001 -- intentional: log & retry any transient failure
            last_err = e
            continue
    return {
        "answer": None,
        "grounded": False,
        "used_chunk_ids": [],
        "latency_ms": 0.0,
        "error": f"generation failed after {max_attempts} attempts: {last_err}",
    }
