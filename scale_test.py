"""
Tests the two open claims from the earlier writeup, with real measurements
instead of assertions:

  1. Does query_type metadata filtering actually improve retrieval when
     plain vector similarity is genuinely ambiguous? (AMBIGUOUS_ENTITIES
     in scale_data.py are built specifically to test this honestly.)
  2. Does vector search latency hold up once the corpus is realistically
     sized (thousands of chunks) instead of the 25-passage toy corpus?
"""

import time
import numpy as np

from scale_data import (
    build_scale_corpus, AMBIGUOUS_EVAL_QUERIES, infer_query_type,
)
from embedding import TfidfEmbedder
from vector_store import VectorStore

SCALE_MULTIPLIER = 60  # ~27 seed passages * 60 = ~1620 chunks; bump this to push scale further

print("Building synthetic corpus at scale...")
passages = build_scale_corpus(scale_multiplier=SCALE_MULTIPLIER)
print(f"  {len(passages)} passages generated\n")

embedder = TfidfEmbedder(dim=128)
embedder.fit([p.text for p in passages])
vectors = embedder.embed_batch([p.text for p in passages])

# vector_store.build() takes Chunk objects (chunk_id/source_passage_id/metadata),
# not Passage objects directly -- wrap each passage as its own single chunk here,
# since we're testing retrieval/filtering mechanics, not chunking strategy.
from chunking import Chunk
chunks = [
    Chunk(chunk_id=p.passage_id, text=p.text, source_passage_id=p.passage_id,
          strategy="whole_passage", position=0, metadata={"topic": p.topic})
    for p in passages
]
chunk_vectors = vectors

store = VectorStore(dim=embedder.dim)
store.build(chunks, chunk_vectors)

print("=" * 78)
print("1. METADATA-AWARE FILTERING: plain similarity vs. query_type-filtered")
print("=" * 78)
print(f"{'query':50s} {'plain top-1 correct?':>22s} {'filtered top-1 correct?':>24s}")
print("-" * 100)

plain_correct = 0
filtered_correct = 0
for query, qtype, expect_substr in AMBIGUOUS_EVAL_QUERIES:
    qvec = embedder.embed(query)

    plain = store.search(qvec, k=1)
    plain_hit = bool(plain["results"]) and expect_substr.lower() in plain["results"][0].text.lower()

    filtered = store.search_with_filter(qvec, k=1, topic=infer_query_type(query))
    filtered_hit = bool(filtered["results"]) and expect_substr.lower() in filtered["results"][0].text.lower()

    plain_correct += plain_hit
    filtered_correct += filtered_hit
    print(f"{query[:48]:50s} {str(plain_hit):>22s} {str(filtered_hit):>24s}")

n = len(AMBIGUOUS_EVAL_QUERIES)
print("-" * 100)
print(f"Plain similarity accuracy   : {plain_correct}/{n} ({100*plain_correct/n:.0f}%)")
print(f"Metadata-filtered accuracy  : {filtered_correct}/{n} ({100*filtered_correct/n:.0f}%)")
print()
if filtered_correct > plain_correct:
    print("=> Metadata filtering resolved cases plain vector similarity got wrong.")
elif filtered_correct == plain_correct:
    print("=> No difference on this query set -- see note below on why, and what would change it.")
else:
    print("=> Filtering did WORSE here -- likely infer_query_type() misclassified a query; "
          "inspect the mismatches above before trusting filtering blindly.")

print()
print("Why this test set doesn't isolate metadata filtering's contribution:")
print("every query above already contains a lexical giveaway ('founded' ->")
print("PERSON/company, 'temperature' -> NUMERIC) that TF-IDF word-overlap")
print("picks up on its own. Real test follows.")
print()

print("=" * 78)
print("1b. GENUINELY AMBIGUOUS QUERIES (identical query text, different intended")
print("    sense per row -- ONLY metadata can possibly disambiguate these)")
print("=" * 78)
from scale_data import GENUINELY_AMBIGUOUS_QUERIES
print(f"{'query':22s} {'forced type':12s} {'plain top-1 correct?':>22s} {'filtered top-1 correct?':>24s}")
print("-" * 85)

plain_correct2 = 0
filtered_correct2 = 0
for query, forced_type, expect_substr in GENUINELY_AMBIGUOUS_QUERIES:
    qvec = embedder.embed(query)

    plain = store.search(qvec, k=1)
    plain_hit = bool(plain["results"]) and expect_substr.lower() in plain["results"][0].text.lower()

    filtered = store.search_with_filter(qvec, k=1, topic=forced_type)
    filtered_hit = bool(filtered["results"]) and expect_substr.lower() in filtered["results"][0].text.lower()

    plain_correct2 += plain_hit
    filtered_correct2 += filtered_hit
    print(f"{query:22s} {forced_type:12s} {str(plain_hit):>22s} {str(filtered_hit):>24s}")

n2 = len(GENUINELY_AMBIGUOUS_QUERIES)
print("-" * 85)
print(f"Plain similarity accuracy   : {plain_correct2}/{n2} ({100*plain_correct2/n2:.0f}%)")
print(f"Metadata-filtered accuracy  : {filtered_correct2}/{n2} ({100*filtered_correct2/n2:.0f}%)")
print()
if filtered_correct2 > plain_correct2:
    print("=> CONFIRMED: metadata filtering resolves cases identical query text")
    print("   cannot, by construction, disambiguate on its own. This is the real")
    print("   case for metadata-aware retrieval, separate from any lexical signal.")
else:
    print("=> Filtering did not help here either -- worth investigating rather")
    print("   than assuming the metadata mechanism works; see printed rows above.")

print()
print("=" * 78)
print(f"2. LATENCY AT SCALE ({len(chunks)} indexed chunks, TF-IDF + real FAISS HNSW)")
print("=" * 78)

test_queries = [q for q, _, _ in AMBIGUOUS_EVAL_QUERIES] * 10  # 70 queries, not a single best-case run

embed_latencies = []
search_latencies = []
for q in test_queries:
    t0 = time.perf_counter()
    qvec = embedder.embed(q)
    embed_latencies.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    store.search(qvec, k=5)
    search_latencies.append((time.perf_counter() - t0) * 1000)

def pct_row(name, values):
    p50, p70, p95, p100 = (np.percentile(values, p) for p in (50, 70, 95, 100))
    print(f"  {name:16s} n={len(values):4d}  P50={p50:6.3f}ms  P70={p70:6.3f}ms  "
          f"P95={p95:6.3f}ms  P100={p100:6.3f}ms")

pct_row("embed_query", embed_latencies)
pct_row("vector_search", search_latencies)
combined = [e + s for e, s in zip(embed_latencies, search_latencies)]
pct_row("combined", combined)

print()
verdict = "MEETS" if max(combined) <= 200.0 else "EXCEEDS"
print(f"Worst-case combined (P100) = {max(combined):.3f}ms -> {verdict} the 200ms target, "
      f"at {len(chunks)}x the toy corpus's scale.")
print()
print("Honest caveats for the writeup:")
print(" - This corpus is SYNTHETIC (paraphrase-prefixed variations of ~27 seed")
print("   passages), not real MSMARCO-XI text. It's a legitimate way to test")
print("   latency scaling (vector count is what drives FAISS search cost, not")
print("   semantic diversity) but recall/accuracy numbers above are only as")
print("   meaningful as the ambiguous-entity design -- re-run this exact script")
print("   against real MSMARCO-XI passages once loaded via load_real_dataset.py")
print("   to get numbers you'd actually cite in a submission.")
print(" - TF-IDF is the embedder here, not the real semantic model you'll ship")
print("   with -- a real sentence-transformer embedding may separate the")
print("   ambiguous-entity senses better (or worse) than TF-IDF's word-overlap")
print("   signal does; this script's filtering-benefit result should be re-")
print("   checked once SentenceTransformerEmbedder is wired in.")
