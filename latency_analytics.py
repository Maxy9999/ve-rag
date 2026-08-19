"""
Stage 9: Latency analytics.

Almost "free" given stage 8's instrumentation -- this module just
aggregates the LatencyLog a harness run already produced into the numbers
the task explicitly asks for (P50/P70/P100), plus P95/P99 as a more
standard companion (P100 == max, and is a single-outlier-sensitive number
worth pairing with something less noisy).

Two entry points:
  - summarize(log)        -> dict of per-stage percentile stats
  - render_report(...)    -> a human-readable text report, ready to paste
                              into a submission writeup or save to a file
"""

import numpy as np


PERCENTILES = [50, 70, 95, 99, 100]


def summarize(log) -> dict:
    """
    log: a LatencyLog (from harness.py), or any object exposing
    .stages() -> list[str] and .values(stage) -> list[float].
    Returns {stage_name: {p50: ..., p70: ..., p95: ..., p99: ..., p100: ...,
                           n: ..., mean: ...}}
    """
    out = {}
    for stage in log.stages():
        values = log.values(stage)
        if not values:
            continue
        stats = {f"p{p}": float(np.percentile(values, p)) for p in PERCENTILES}
        stats["n"] = len(values)
        stats["mean"] = float(np.mean(values))
        out[stage] = stats
    return out


def end_to_end_summary(total_latencies_ms: list[float]) -> dict:
    if not total_latencies_ms:
        return {}
    stats = {f"p{p}": float(np.percentile(total_latencies_ms, p)) for p in PERCENTILES}
    stats["n"] = len(total_latencies_ms)
    stats["mean"] = float(np.mean(total_latencies_ms))
    return stats


def render_report(stage_summary: dict, end_to_end: dict, target_ms: float = 200.0,
                   notes: str | None = None) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("LATENCY REPORT")
    lines.append("=" * 78)
    lines.append(f"{'stage':22s} {'n':>5s} {'mean':>9s} {'P50':>9s} {'P70':>9s} "
                  f"{'P95':>9s} {'P99':>9s} {'P100':>9s}")
    lines.append("-" * 78)
    for stage, s in stage_summary.items():
        lines.append(
            f"{stage:22s} {s['n']:5d} {s['mean']:8.2f}m {s['p50']:8.2f}m "
            f"{s['p70']:8.2f}m {s['p95']:8.2f}m {s['p99']:8.2f}m {s['p100']:8.2f}m"
        )
    lines.append("-" * 78)
    if end_to_end:
        s = end_to_end
        lines.append(
            f"{'END_TO_END':22s} {s['n']:5d} {s['mean']:8.2f}m {s['p50']:8.2f}m "
            f"{s['p70']:8.2f}m {s['p95']:8.2f}m {s['p99']:8.2f}m {s['p100']:8.2f}m"
        )
        lines.append("-" * 78)
        verdict = "MEETS" if s["p100"] <= target_ms else "EXCEEDS"
        lines.append(f"Target: {target_ms:.0f}ms end-to-end.  Worst observed (P100): "
                      f"{s['p100']:.2f}ms -> {verdict} target.")
    lines.append("=" * 78)
    if notes:
        lines.append(notes)
    return "\n".join(lines)


def breakdown_share(stage_summary: dict, metric: str = "p50") -> dict:
    """
    What fraction of total (summed-stage) latency does each stage account
    for, at a given percentile. Useful for pointing at the real bottleneck
    rather than guessing -- e.g. "generation is 92% of the budget at P50."
    """
    total = sum(s[metric] for s in stage_summary.values())
    if total == 0:
        return {stage: 0.0 for stage in stage_summary}
    return {stage: s[metric] / total for stage, s in stage_summary.items()}
