"""
Runs and prints results for everything buildable/testable in this sandbox:

  A. Chunking strategy comparison        (stage 1)
  B. Retrieval recall@K on labeled pairs (stage 2 + 3 correctness)
  C. Guardrail demo                      (stage 4 + 6, in-domain vs off-topic vs hallucination)
  D. Latency analytics P50/P70/P95/P100  (stages 2,3,5,6 individually + end-to-end)
"""

import time
import numpy as np

from sample_data import load_passages, load_eval_pairs, load_offtopic_queries
from chunking import run_all_strategies
from embedding import TfidfEmbedder
from vector_store import VectorStore
from harness_partial import PartialRAGHarness

passages = load_passages()
eval_pairs = load_eval_pairs()
offtopic_queries = load_offtopic_queries()

print("=" * 70)
print("A. CHUNKING STRATEGY COMPARISON")
print("=" * 70)

all_chunks = run_all_strategies(passages)
for strategy_name, chunks in all_chunks.items():
    n = len(chunks)
    avg_len = np.mean([len(c.text) for c in chunks])
    print(f"  {strategy_name:28s}  n_chunks={n:4d}   avg_chunk_chars={avg_len:6.1f}")

print()
print("Interpretation: naive_fixed produces the most chunks (mid-word/")
print("mid-sentence cuts) -- more fragments to index, and each fragment is")
print("semantically incomplete. sentence_aware and whole_passage stay close")
print("to 1 chunk/passage here because MSMARCO-style passages are already")
print("short & atomic -- splitting them further doesn't help. This is the")
print("kind of empirical justification the task asks for, run against your")
print("real corpus stats once you load actual MSMARCO-XI data.")
print()


def recall_at_k(chunks, k=3, min_score=None):
    """
    For each (query, correct_passage_id) pair: embed+index this chunk set,
    retrieve top-k, check whether the correct passage's chunk is present.
    """
    embedder = TfidfEmbedder(dim=64)
    embedder.fit([c.text for c in chunks])
    vecs = embedder.embed_batch([c.text for c in chunks])
    store = VectorStore(dim=embedder.dim)
    store.build(chunks, vecs)

    hits = 0
    for query, correct_pid in eval_pairs:
        qvec = embedder.embed(query)
        out = store.search(qvec, k=k)
        retrieved_pids = {r.source_passage_id for r in out["results"]}
        if correct_pid in retrieved_pids:
            hits += 1
    return hits / len(eval_pairs)


print("=" * 70)
print("B. RETRIEVAL RECALL@3 PER CHUNKING STRATEGY")
print("   (fraction of test queries where the correct passage was in top-3)")
print("=" * 70)
for strategy_name, chunks in all_chunks.items():
    r = recall_at_k(chunks, k=3)
    print(f"  {strategy_name:28s}  recall@3 = {r:.3f}  ({int(r*len(eval_pairs))}/{len(eval_pairs)})")
print()

print("=" * 70)
print("C. GUARDRAIL DEMO")
print("=" * 70)

harness = PartialRAGHarness(passages, hallucination_rate=0.0, simulate_llm_latency=False)

print("\n-- In-domain queries (should be answered) --")
for q, _ in eval_pairs[:3]:
    t = harness.answer(q)
    print(f"  [{t['final_status']:20s}] {q}")

print("\n-- Off-topic queries (should be refused, pre-generation) --")
for q in offtopic_queries:
    t = harness.answer(q, min_score=0.15)
    print(f"  [{t['final_status']:20s}] {q}  (best_score={t['retrieved'][0][1] if t['retrieved'] else None})")

print("\n-- Forced-hallucination demo (should be refused, post-generation) --")
harness_halluc = PartialRAGHarness(passages, hallucination_rate=1.0, simulate_llm_latency=False)
for q, _ in eval_pairs[:3]:
    t = harness_halluc.answer(q)
    print(f"  [{t['final_status']:20s}] {q}")
print()

print("=" * 70)
print("D. LATENCY ANALYTICS (P50 / P70 / P95 / P100), across", len(eval_pairs), "queries")
print("   (simulated realistic LLM latency ON for this run)")
print("=" * 70)

harness_timed = PartialRAGHarness(passages, hallucination_rate=0.1, simulate_llm_latency=True)
end_to_end_ms = []
for q, _ in eval_pairs:
    start = time.perf_counter()
    harness_timed.answer(q)
    end_to_end_ms.append((time.perf_counter() - start) * 1000)

def pct_table(name, values):
    p50 = np.percentile(values, 50)
    p70 = np.percentile(values, 70)
    p95 = np.percentile(values, 95)
    p100 = np.percentile(values, 100)
    print(f"  {name:20s}  P50={p50:7.2f}ms  P70={p70:7.2f}ms  P95={p95:7.2f}ms  P100={p100:7.2f}ms")

for stage in harness_timed.log.stages():
    pct_table(stage, harness_timed.log.values(stage))
pct_table("END_TO_END (2-6)", end_to_end_ms)

print()
print("Note: 'generation' dominates because MockLLM simulates realistic")
print("hosted-LLM latency (150-400ms sleep) on purpose. embed_query and")
print("vector_search stay in low single-digit ms even with simulated")
print("latency off elsewhere -- that's the part of the budget under our")
print("control. STT (stage 7, not wired in yet) will typically add another")
print("300ms-2s on top of this when real audio is involved -- see harness")
print("design notes on how the 200ms target is interpreted.")
