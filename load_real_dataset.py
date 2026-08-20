"""
Loads a slice of the real ai4bharat/MSMARCO-XI dataset and reshapes it
into the same Passage / eval-pairs shape sample_data.py's toy corpus uses.
Caches the result to disk so you don't re-download/re-flatten every run.

METADATA MAPPING NOTE:
MS MARCO's `query_type` field (inherited as-is by MSMARCO-XI) is one of a
fixed, known set of 5 categories, not a free-form string:
    DESCRIPTION  - "how does X work", "what causes Y"
    NUMERIC      - "how many", "what year", "how much"
    ENTITY       - "what is the name of..."
    LOCATION     - "where is X"
    PERSON       - "who is X"
Every passage attached to a query inherits that query's query_type as its
`topic` here. This is a real, meaningful metadata field (unlike the
placeholder "general" the toy corpus used) -- e.g. you can now filter
retrieval to LOCATION-type passages when the question is clearly asking
"where", which disambiguates cases where plain vector similarity alone
picks the wrong passage among several plausible ones. See
scale_test.py for a concrete before/after demonstration of this.
"""
import json
import os
from datasets import load_dataset
from sample_data import Passage

CACHE_DIR = "data_cache"
LANGUAGE = "hi"       # Hindi -- best model/tooling support of the 14 options
N_ROWS = 5000          # arbitrary, chosen for fast iteration -- see conversation
                       # notes on when to raise this for real submission numbers

# File names in the repo do NOT follow a consistent {code}train.parquet
# pattern -- confirmed by a real FileNotFoundError looking for
# "hitrain.parquet" (wrong) when the actual file is "hintrain.parquet".
# The dataset card documents the real per-language file prefix in a table;
# it's NOT simply the 2-letter language code (e.g. "hi" -> "hin", "te" ->
# "tel", but "gu" -> "gu" and "or" -> "or" unchanged). Transcribed directly
# from that table rather than inferred, since the pattern isn't regular.
FILE_PREFIX = {
    "as": "asm", "bn": "ben", "gu": "gu", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "or", "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
}


def _cache_paths():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return (
        os.path.join(CACHE_DIR, f"passages_{LANGUAGE}.json"),
        os.path.join(CACHE_DIR, f"eval_pairs_{LANGUAGE}.json"),
    )


def build_and_cache():
    passages_path, eval_path = _cache_paths()
    if os.path.exists(passages_path) and os.path.exists(eval_path):
        return  # already built, reuse cache

    # NOTE: the dataset card's documented usage --
    #   load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train")
    # -- does NOT work against the actual repo: it only exposes a single
    # "default" config (confirmed via a real ValueError: "BuilderConfig
    # 'hi' not found. Available: ['default']"). The repo's README YAML
    # doesn't register per-language configs even though the files
    # themselves ARE split by language (train/hintrain.parquet,
    # validation/hinval.parquet, etc.) -- so we point data_files directly
    # at the language-specific parquet file instead of relying on a named
    # config that isn't actually registered.
    #
    # train/hintrain.parquet alone is ~3.72GB. streaming=True avoids
    # downloading the whole file just to keep the first 5000 rows --
    # .take() stops reading once satisfied instead.
    file_prefix = FILE_PREFIX[LANGUAGE]
    ds = load_dataset("parquet", 
        data_files={"train": "hf_cache/hintrain.parquet"}, 
        split="train", 
        streaming=True).take(5000)

    seen = {}          # passage text -> passage_id (dedupe across query rows)
    passages = []
    eval_pairs = []

    for row in ds:
        translated = row["passages"]["Translated_passages"]
        selected = row["passages"]["is_selected"]
        # query_type is one of DESCRIPTION / NUMERIC / ENTITY / LOCATION / PERSON
        query_type = row.get("query_type", "DESCRIPTION")

        correct_pid = None
        for text, is_sel in zip(translated, selected):
            if text not in seen:
                pid = f"p{len(passages)}"
                seen[text] = pid
                passages.append({
                    "passage_id": pid, "text": text,
                    "language": LANGUAGE, "topic": query_type,
                })
            if is_sel == 1 and correct_pid is None:
                correct_pid = seen[text]

        if correct_pid is not None:
            eval_pairs.append([row["query"], correct_pid, query_type])

    with open(passages_path, "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_pairs, f, ensure_ascii=False)


def load_real_passages():
    build_and_cache()
    passages_path, _ = _cache_paths()
    with open(passages_path, encoding="utf-8") as f:
        raw = json.load(f)
    return [Passage(**p) for p in raw]


def load_real_eval_pairs(with_query_type=False):
    build_and_cache()
    _, eval_path = _cache_paths()
    with open(eval_path, encoding="utf-8") as f:
        raw = json.load(f)
    if with_query_type:
        return [tuple(p) for p in raw]           # (query, passage_id, query_type)
    return [(p[0], p[1]) for p in raw]            # (query, passage_id) -- matches toy corpus shape
