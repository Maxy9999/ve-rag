"""
Stages 4 & 6: Guardrails.

Two checkpoints:
  - pre_generation_check   : runs BEFORE calling the LLM. Uses retrieval
                              scores we already computed in stage 3 -- so
                              this check is nearly free (no extra API call,
                              no extra latency of consequence).
  - post_generation_check  : runs AFTER the LLM answers. Verifies the
                              answer is actually grounded in the retrieved
                              context, not hallucinated. Has a cheap mode
                              (trust the model's self-reported `grounded`
                              flag) and a rigorous mode (a second LLM call
                              purely to verify -- costs latency, catches
                              more).

Both return a structured GuardrailResult so the harness can log WHY a
query was refused (off_topic / low_confidence / not_grounded / ok) --
that per-reason logging is itself a submission artifact worth showing.
"""

from dataclasses import dataclass


@dataclass
class GuardrailResult:
    passed: bool
    reason: str  # "ok" | "off_topic" | "low_similarity" | "not_grounded" | "low_stt_confidence"
    detail: str = ""


def stt_confidence_check(stt_output: dict, min_confidence: float = 0.6) -> GuardrailResult:
    """
    Runs immediately after STT, before anything downstream even sees the
    transcript. If the STT provider's own confidence score is too low,
    refuse early rather than feeding a probably-mistranscribed question
    into retrieval (garbage in, garbage out -- and it wastes the retrieval
    + generation latency budget on a query we already suspect is wrong).
    """
    conf = stt_output.get("confidence")
    if stt_output.get("error"):
        return GuardrailResult(False, "low_stt_confidence", f"STT error: {stt_output['error']}")
    if conf is not None and conf < min_confidence:
        return GuardrailResult(
            False, "low_stt_confidence",
            f"STT confidence {conf:.2f} below threshold {min_confidence}"
        )
    return GuardrailResult(True, "ok")


def pre_generation_check(search_results, min_score: float = 0.35) -> GuardrailResult:
    """
    If the single best retrieved chunk doesn't clear a similarity threshold,
    there's nothing relevant in the dataset for this query -- refuse before
    ever calling the (slower, costlier) generation stage.
    """
    if not search_results:
        return GuardrailResult(False, "off_topic", "no chunks retrieved")

    best = max(search_results, key=lambda r: r.score)
    if best.score < min_score:
        return GuardrailResult(
            False, "low_similarity",
            f"best score {best.score:.3f} below threshold {min_score}"
        )
    return GuardrailResult(True, "ok")


def post_generation_check_selfreport(generation_output: dict) -> GuardrailResult:
    """
    CHEAP mode: trust the LLM's own structured `grounded` field (see
    generation.py -- we ask the model to self-report this in the same
    call that produces the answer, no extra latency).
    """
    if not generation_output.get("grounded", False):
        return GuardrailResult(
            False, "not_grounded",
            "model self-reported the answer is not supported by context"
        )
    return GuardrailResult(True, "ok")


def post_generation_check_nli(answer: str, context_chunks: list[str], nli_fn) -> GuardrailResult:
    """
    RIGOROUS mode: a second, independent check -- pass (context, answer) to
    an entailment function (`nli_fn`, e.g. a second LLM call asking
    "does this context support this answer? yes/no", or a real NLI model)
    and only pass if it confirms entailment. Costs an extra call's worth of
    latency -- that tradeoff should be stated explicitly in the latency
    report (this is why we keep it as a separate, optional stage rather
    than folding it into the cheap check).
    """
    combined_context = " ".join(context_chunks)
    verdict = nli_fn(context=combined_context, answer=answer)
    if not verdict:
        return GuardrailResult(False, "not_grounded", "NLI check: context does not entail answer")
    return GuardrailResult(True, "ok")
