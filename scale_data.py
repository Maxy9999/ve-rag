"""
SYNTHETIC stand-in for a MSMARCO-XI-scale corpus, used because this
sandbox can't reach huggingface.co. Unlike sample_data.py's 25-passage
toy corpus, this one is built to actually stress two specific claims:

  1. Does metadata-aware filtering (by query_type: DESCRIPTION / NUMERIC /
     ENTITY / LOCATION / PERSON) improve retrieval over plain vector
     similarity? To test this honestly we need passages where similarity
     ALONE is genuinely ambiguous -- so this corpus deliberately includes
     entities with multiple unrelated senses (e.g. "Amazon" the river vs
     "Amazon" the company), each tagged with a different query_type. If
     filtering doesn't help here, it wouldn't just be "unproven" -- we'd
     have caught it not working.

  2. Does latency hold up once the corpus is thousands of chunks instead
     of 25? Generates SCALE_MULTIPLIER paraphrase-style variations of a
     base set of passages to reach a realistic scale.
"""

from dataclasses import dataclass
import random

from sample_data import Passage

# Entities with genuinely different senses under different query_types --
# this is what makes the metadata-filtering test meaningful rather than
# rigged. Vector similarity on "Amazon" alone can't tell these apart;
# query_type can.
AMBIGUOUS_ENTITIES = [
    ("Amazon", "LOCATION", "The Amazon river carries more water than the next seven largest rivers combined, draining a basin that spans eight countries in South America."),
    ("Amazon", "ENTITY", "Amazon is an e-commerce and cloud computing company that began as an online bookstore before expanding into AWS, logistics, and streaming."),
    ("Amazon", "PERSON", "Jeff Bezos founded Amazon in 1994 in a garage in Bellevue, Washington, initially selling books before expanding the catalog."),
    ("Mercury", "LOCATION", "Mercury is the closest planet to the Sun and has no moons, completing an orbit in just 88 Earth days."),
    ("Mercury", "ENTITY", "Mercury is a heavy silvery liquid metal at room temperature, historically used in thermometers and barometers."),
    ("Mercury", "NUMERIC", "Mercury's surface temperature ranges from about -173°C at night to 427°C during the day, the widest swing of any planet."),
    ("Jordan", "LOCATION", "Jordan is a country in the Middle East bordered by Saudi Arabia, Iraq, Syria, and Israel, with the Dead Sea along its western edge."),
    ("Jordan", "PERSON", "Michael Jordan won six NBA championships with the Chicago Bulls and is widely regarded as one of basketball's greatest players."),
    ("Turkey", "LOCATION", "Turkey straddles both Europe and Asia, separated by the Bosphorus strait that runs through Istanbul."),
    ("Turkey", "ENTITY", "A turkey is a large bird native to North America, commonly raised for its meat and traditionally served at Thanksgiving."),
]

# Base topic passages (same as sample_data.py's 25, kept here so this file
# is self-contained) plus their natural query_type classification, used as
# the seed set that gets expanded into thousands of variations.
BASE_PASSAGES = [
    ("networking", "DESCRIPTION", "TCP is a connection-oriented protocol that guarantees reliable, ordered delivery of data between applications over an IP network."),
    ("networking", "DESCRIPTION", "UDP is a connectionless protocol that sends packets without establishing a session first, trading reliability for lower latency."),
    ("networking", "DESCRIPTION", "A firewall inspects incoming and outgoing network traffic and blocks packets that do not match a defined set of security rules."),
    ("networking", "DESCRIPTION", "DNS translates human-readable domain names into the IP addresses that computers use to identify each other on a network."),
    ("networking", "NUMERIC", "A standard IPv4 subnet mask uses 32 bits total, commonly split as 24 bits for the network portion and 8 for the host portion in a /24 network."),
    ("cooking", "DESCRIPTION", "Blanching vegetables briefly in boiling water and then plunging them into ice water halts the cooking process and preserves their bright color."),
    ("cooking", "DESCRIPTION", "Deglazing a pan means adding liquid to lift the browned bits stuck to the bottom after searing meat, forming the base of a flavorful sauce."),
    ("cooking", "NUMERIC", "Bread dough typically proofs for 1 to 2 hours at room temperature, or up to 24 hours if retarded slowly in the refrigerator."),
    ("geography", "LOCATION", "The Ganges river originates in the Himalayas and flows across northern India before draining into the Bay of Bengal."),
    ("geography", "LOCATION", "Goa is the smallest state in India by area, located on the southwestern coast along the Arabian Sea."),
    ("geography", "NUMERIC", "The Thar Desert spans roughly 200,000 square kilometers across northwestern India and eastern Pakistan."),
    ("biology", "DESCRIPTION", "Mitochondria generate most of a cell's ATP through oxidative phosphorylation, which is why they are often called the powerhouse of the cell."),
    ("biology", "DESCRIPTION", "Photosynthesis converts carbon dioxide and water into glucose and oxygen using light energy captured by chlorophyll in plant cells."),
    ("biology", "PERSON", "James Watson and Francis Crick are credited with determining the double-helix structure of DNA in 1953, building on Rosalind Franklin's X-ray data."),
    ("finance", "DESCRIPTION", "Compound interest is calculated on both the initial principal and the accumulated interest from previous periods, causing balances to grow faster over time."),
    ("finance", "NUMERIC", "A credit score in most US scoring models ranges from 300 to 850, with scores above 700 generally considered good."),
    ("finance", "ENTITY", "A mutual fund pools money from many investors to buy a diversified portfolio of stocks, bonds, or other securities, managed by a professional."),
]

VARIATION_PREFIXES = [
    "", "According to standard references, ", "In technical terms, ",
    "As commonly explained, ", "Put simply, ", "Broadly speaking, ",
    "In most textbooks, ", "Generally, ",
]

VARIATION_SUFFIXES = [
    "", " This is a well-established fact.", " Sources generally agree on this point.",
    " This detail is often cited in overviews.", " Reference material confirms this.",
]
# 8 prefixes x 5 suffixes = 40 combinations -- at scale_multiplier=60 this
# still produces ~1.5x duplication per seed passage rather than ~7.5x,
# which is close enough to what real (imperfectly-deduplicated) corpora
# look like to be a fair latency/HNSW-robustness test. Found via testing:
# 8 prefixes alone (7.5x duplication) was dense enough to break FAISS
# HNSW's default efConstruction=40 -- see vector_store.py's fix + note.


def build_scale_corpus(scale_multiplier: int = 60, seed: int = 0):
    """
    Expands BASE_PASSAGES + AMBIGUOUS_ENTITIES into a corpus of roughly
    (len(BASE_PASSAGES) + len(AMBIGUOUS_ENTITIES)) * scale_multiplier
    passages, by prefixing light paraphrase variations. This is NOT a
    substitute for real corpus diversity (see honest caveat in
    scale_test.py's output) -- it's a controlled way to test latency and
    metadata-filtering at realistic vector-count scale without needing
    the real dataset.
    """
    rng = random.Random(seed)
    passages = []
    pid_counter = 0

    all_seed = [(topic, qtype, text) for topic, qtype, text in BASE_PASSAGES]
    all_seed += [(f"entity:{name}", qtype, text) for name, qtype, text in AMBIGUOUS_ENTITIES]

    for topic, qtype, text in all_seed:
        for _ in range(scale_multiplier):
            prefix = rng.choice(VARIATION_PREFIXES)
            suffix = rng.choice(VARIATION_SUFFIXES)
            passages.append(Passage(
                passage_id=f"p{pid_counter}",
                text=prefix + text + suffix,
                language="en",
                topic=qtype,
            ))
            pid_counter += 1

    return passages


# These queries contain their own disambiguating vocabulary ("founded",
# "temperature") -- TF-IDF word-overlap alone can solve them, which means
# they DON'T isolate what metadata filtering specifically contributes.
# Kept as a first pass / sanity check, not the real test -- see
# GENUINELY_AMBIGUOUS_QUERIES below for that.
AMBIGUOUS_EVAL_QUERIES = [
    # (query, inferred_query_type, expected_answer_substring)
    ("Where does the Amazon river flow through?", "LOCATION", "river"),
    ("Who founded Amazon?", "PERSON", "Bezos"),
    ("Where is Mercury located relative to the Sun?", "LOCATION", "planet"),
    ("What temperature swings does Mercury experience?", "NUMERIC", "temperature"),
    ("Where is the country of Jordan located?", "LOCATION", "Middle East"),
    ("Who is Michael Jordan?", "PERSON", "basketball"),
    ("Where does Turkey sit geographically?", "LOCATION", "Europe and Asia"),
]

# The REAL test: bare "tell me about X" queries that carry NO lexical hint
# of which sense of the ambiguous entity is meant -- only the (externally
# supplied) query_type distinguishes them. This is what isolates metadata
# filtering's actual contribution, separate from whatever word-overlap
# signal the query happens to leak on its own.
GENUINELY_AMBIGUOUS_QUERIES = [
    # (query, forced_query_type, expected_answer_substring)
    ("Tell me about Amazon.", "LOCATION", "river"),
    ("Tell me about Amazon.", "ENTITY", "e-commerce"),
    ("Tell me about Amazon.", "PERSON", "Bezos"),
    ("Tell me about Mercury.", "LOCATION", "planet"),
    ("Tell me about Mercury.", "ENTITY", "liquid metal"),
    ("Tell me about Jordan.", "LOCATION", "Middle East"),
    ("Tell me about Jordan.", "PERSON", "basketball"),
    ("Tell me about Turkey.", "LOCATION", "Europe and Asia"),
    ("Tell me about Turkey.", "ENTITY", "bird"),
]


def infer_query_type(query: str) -> str:
    """
    Simple heuristic query_type classifier -- stands in for a real
    classifier or the LLM itself tagging query_type at generation time.
    Good enough to demonstrate the filtering mechanism; not a serious
    NLP component.
    """
    q = query.lower()
    if q.startswith("who"):
        return "PERSON"
    if q.startswith("where"):
        return "LOCATION"
    if any(w in q for w in ["how many", "how much", "what year", "what temperature"]):
        return "NUMERIC"
    if q.startswith("what is the name") or q.startswith("what is a") or q.startswith("what is"):
        return "ENTITY"
    return "DESCRIPTION"
