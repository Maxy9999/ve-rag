"""
Stage 1: Chunking.

Deliberately implements more than one strategy, per the task requirement
("chunking strategy should be vast -- not a single naive fixed-size approach").

Each strategy takes a list of `Passage` objects (see sample_data.py) and
returns a list of `Chunk` objects. A Chunk always carries metadata back to
its source passage -- this is what makes retrieval results explainable
("this answer came from passage p13, chunk 2") and is the hook for
metadata-aware filtering later (e.g. restrict retrieval to one topic/language).

MSMARCO passages are usually short (1-3 sentences) already, so for THIS
dataset shape, sentence-aware / whole-passage chunking will often be the
practically-best strategy -- but we still implement fixed-size+overlap and
a naive-baseline so we can empirically justify that choice with numbers
rather than assert it (see test_chunking.py's comparison).
"""

from dataclasses import dataclass, field
import re


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_passage_id: str
    strategy: str
    position: int  # index of this chunk within its source passage
    metadata: dict = field(default_factory=dict)


def _split_sentences(text: str) -> list[str]:
    # Lightweight sentence splitter -- good enough for MSMARCO-style short
    # passages. Swap for a proper sentence tokenizer (e.g. nltk.sent_tokenize
    # or a spaCy pipeline) on the real dataset if passages get longer/messier.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def chunk_naive_fixed(passages, chunk_size_chars=80):
    """
    BASELINE (explicitly the thing the task says NOT to submit alone).
    Splits raw text every N characters, no regard for word/sentence
    boundaries, no overlap. Kept here only so we can show, with numbers,
    why it's worse than the alternatives below.
    """
    chunks = []
    for p in passages:
        text = p.text
        for i, start in enumerate(range(0, len(text), chunk_size_chars)):
            piece = text[start:start + chunk_size_chars]
            chunks.append(Chunk(
                chunk_id=f"{p.passage_id}_naive_{i}",
                text=piece,
                source_passage_id=p.passage_id,
                strategy="naive_fixed",
                position=i,
                metadata={"topic": p.topic, "language": p.language},
            ))
    return chunks


def chunk_fixed_with_overlap(passages, chunk_size_words=25, overlap_words=5):
    """
    Fixed-size chunking, but by WORD count (not raw characters, so we don't
    slice mid-word) and with overlap between consecutive chunks, so a
    concept that straddles a chunk boundary isn't lost entirely to either
    chunk. This is the standard "reasonable default" strategy.
    """
    chunks = []
    for p in passages:
        words = p.text.split()
        if not words:
            continue
        step = max(1, chunk_size_words - overlap_words)
        i = 0
        pos = 0
        while i < len(words):
            piece_words = words[i:i + chunk_size_words]
            chunks.append(Chunk(
                chunk_id=f"{p.passage_id}_fixed_{pos}",
                text=" ".join(piece_words),
                source_passage_id=p.passage_id,
                strategy="fixed_overlap",
                position=pos,
                metadata={"topic": p.topic, "language": p.language,
                          "overlap_words": overlap_words},
            ))
            if i + chunk_size_words >= len(words):
                break
            i += step
            pos += 1
    return chunks


def chunk_sentence_aware(passages, max_sentences_per_chunk=2):
    """
    Groups whole sentences together up to a cap, never cutting a sentence
    in half. For short, already-coherent MSMARCO-style passages this often
    means "one passage == one chunk", which is usually correct: MSMARCO
    passages are curated to already be a single coherent unit of meaning.
    """
    chunks = []
    for p in passages:
        sentences = _split_sentences(p.text)
        if not sentences:
            continue
        for pos, i in enumerate(range(0, len(sentences), max_sentences_per_chunk)):
            group = sentences[i:i + max_sentences_per_chunk]
            chunks.append(Chunk(
                chunk_id=f"{p.passage_id}_sent_{pos}",
                text=" ".join(group),
                source_passage_id=p.passage_id,
                strategy="sentence_aware",
                position=pos,
                metadata={"topic": p.topic, "language": p.language,
                          "n_sentences": len(group)},
            ))
    return chunks


def chunk_whole_passage_metadata_aware(passages):
    """
    Treats each passage as a single chunk (no splitting at all), but attaches
    rich metadata. This is included because MSMARCO passages are short enough
    that splitting them can actively HURT retrieval (you fragment an already-
    atomic unit of meaning). Metadata-aware here means: every chunk carries
    enough structured info (topic, language, length) that retrieval can
    later filter/boost on it, not just do pure vector similarity.
    """
    chunks = []
    for p in passages:
        chunks.append(Chunk(
            chunk_id=f"{p.passage_id}_whole",
            text=p.text,
            source_passage_id=p.passage_id,
            strategy="whole_passage_metadata",
            position=0,
            metadata={
                "topic": p.topic,
                "language": p.language,
                "char_length": len(p.text),
                "word_count": len(p.text.split()),
            },
        ))
    return chunks


STRATEGIES = {
    "naive_fixed": chunk_naive_fixed,
    "fixed_overlap": chunk_fixed_with_overlap,
    "sentence_aware": chunk_sentence_aware,
    "whole_passage_metadata": chunk_whole_passage_metadata_aware,
}


def run_all_strategies(passages) -> dict[str, list[Chunk]]:
    return {name: fn(passages) for name, fn in STRATEGIES.items()}
