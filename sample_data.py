"""
Synthetic stand-in for ai4bharat/MSMARCO-XI.

Real MSMARCO-style datasets have this shape (per row):
    - passage_id  : unique id for a text passage
    - passage     : the passage text (a paragraph, typically 1-3 sentences)
    - query       : a natural-language question associated with the passage
    - query_id    : unique id for the query
    - is_relevant : whether this (query, passage) pair is a true match (1) or a
                    sampled negative (0) -- MSMARCO ships hard negatives for
                    exactly this purpose: evaluating retrieval quality.

On your machine, replace `load_passages()` / `load_eval_pairs()` below with:

    from datasets import load_dataset
    ds = load_dataset("ai4bharat/MSMARCO-XI")

and reshape into the same PASSAGES / EVAL_PAIRS structures so nothing else
in the pipeline needs to change.
"""

from dataclasses import dataclass


@dataclass
class Passage:
    passage_id: str
    text: str
    # Metadata-aware chunking hook: MSMARCO-XI rows carry a language tag and
    # a source query grouping -- we keep placeholders for both.
    language: str = "en"
    topic: str = "general"


# A small multi-topic corpus so retrieval has something real to discriminate
# between (topics: networking, cooking, geography, biology, finance).
RAW_PASSAGES = [
    ("p1", "networking", "TCP is a connection-oriented protocol that guarantees reliable, "
     "ordered delivery of data between applications over an IP network."),
    ("p2", "networking", "UDP is a connectionless protocol that sends packets without "
     "establishing a session first, trading reliability for lower latency."),
    ("p3", "networking", "A firewall inspects incoming and outgoing network traffic and "
     "blocks packets that do not match a defined set of security rules."),
    ("p4", "networking", "DNS translates human-readable domain names into the IP addresses "
     "that computers use to identify each other on a network."),
    ("p5", "networking", "A subnet mask divides an IP address into a network portion and a "
     "host portion, determining which addresses belong to the same local network."),

    ("p6", "cooking", "Blanching vegetables briefly in boiling water and then plunging them "
     "into ice water halts the cooking process and preserves their bright color."),
    ("p7", "cooking", "Deglazing a pan means adding liquid to lift the browned bits stuck to "
     "the bottom after searing meat, forming the base of a flavorful sauce."),
    ("p8", "cooking", "Proofing bread dough allows yeast to ferment sugars and release carbon "
     "dioxide, which makes the dough rise before baking."),
    ("p9", "cooking", "Tempering chocolate involves carefully controlling its temperature so "
     "the cocoa butter crystallizes into a stable form, giving it a glossy snap."),
    ("p10", "cooking", "Searing meat at high heat browns the surface through the Maillard "
     "reaction, developing flavor, though it does not actually seal in juices."),

    ("p11", "geography", "The Ganges river originates in the Himalayas and flows across "
     "northern India before draining into the Bay of Bengal."),
    ("p12", "geography", "The Deccan Plateau covers much of southern India and is bordered by "
     "the Western Ghats and Eastern Ghats mountain ranges."),
    ("p13", "geography", "Goa is the smallest state in India by area, located on the "
     "southwestern coast along the Arabian Sea."),
    ("p14", "geography", "The Thar Desert spans parts of northwestern India and eastern "
     "Pakistan, receiving very low annual rainfall."),
    ("p15", "geography", "The Western Ghats are a mountain range running along the western "
     "coast of India and are recognized as a UNESCO biodiversity hotspot."),

    ("p16", "biology", "Mitochondria generate most of a cell's ATP through oxidative "
     "phosphorylation, which is why they are often called the powerhouse of the cell."),
    ("p17", "biology", "Photosynthesis converts carbon dioxide and water into glucose and "
     "oxygen using light energy captured by chlorophyll in plant cells."),
    ("p18", "biology", "DNA replication is semi-conservative, meaning each new double helix "
     "contains one original strand and one newly synthesized strand."),
    ("p19", "biology", "Enzymes are biological catalysts that speed up reactions by lowering "
     "the activation energy required, without being consumed themselves."),
    ("p20", "biology", "The blood-brain barrier is a selective membrane that restricts which "
     "substances can pass from the bloodstream into brain tissue."),

    ("p21", "finance", "Compound interest is calculated on both the initial principal and the "
     "accumulated interest from previous periods, causing balances to grow faster over time."),
    ("p22", "finance", "A mutual fund pools money from many investors to buy a diversified "
     "portfolio of stocks, bonds, or other securities, managed by a professional."),
    ("p23", "finance", "Inflation erodes purchasing power over time, meaning the same amount "
     "of money buys fewer goods and services as prices rise."),
    ("p24", "finance", "A credit score summarizes a borrower's creditworthiness based on "
     "repayment history, debt levels, and length of credit history."),
    ("p25", "finance", "Diversification spreads investment risk across different asset classes "
     "so that a loss in one area does not sink the entire portfolio."),
]

PASSAGES = [Passage(passage_id=pid, text=text, topic=topic) for pid, topic, text in RAW_PASSAGES]

# (query, correct_passage_id) pairs -- our "ground truth" for measuring
# retrieval recall@K, standing in for MSMARCO's labeled query-passage pairs.
EVAL_PAIRS = [
    ("What does TCP guarantee when sending data?", "p1"),
    ("Why is UDP faster than TCP?", "p2"),
    ("What does a firewall do to network traffic?", "p3"),
    ("How does DNS work?", "p4"),
    ("What is the purpose of a subnet mask?", "p5"),
    ("Why do you plunge blanched vegetables into ice water?", "p6"),
    ("What does deglazing a pan mean?", "p7"),
    ("Why does bread dough rise during proofing?", "p8"),
    ("What happens when you temper chocolate?", "p9"),
    ("Does searing meat actually seal in the juices?", "p10"),
    ("Where does the Ganges river originate?", "p11"),
    ("What mountain ranges border the Deccan Plateau?", "p12"),
    ("Which is the smallest state in India by area?", "p13"),
    ("What is the climate like in the Thar Desert?", "p14"),
    ("What is special about the Western Ghats?", "p15"),
    ("Why are mitochondria called the powerhouse of the cell?", "p16"),
    ("What does photosynthesis produce?", "p17"),
    ("What does semi-conservative DNA replication mean?", "p18"),
    ("How do enzymes speed up reactions?", "p19"),
    ("What does the blood-brain barrier restrict?", "p20"),
    ("How is compound interest different from simple interest?", "p21"),
    ("What is a mutual fund?", "p22"),
    ("How does inflation affect purchasing power?", "p23"),
    ("What factors go into a credit score?", "p24"),
    ("Why is diversification useful in investing?", "p25"),
]

# A handful of deliberately off-topic / unanswerable queries for guardrail testing.
OFFTOPIC_QUERIES = [
    "What is the capital of France?",
    "Who won the last World Cup?",
    "Write me a poem about the ocean.",
    "What's the weather going to be like tomorrow?",
]


def load_passages():
    return PASSAGES


def load_eval_pairs():
    return EVAL_PAIRS


def load_offtopic_queries():
    return OFFTOPIC_QUERIES
