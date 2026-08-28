"""Offline separation analysis (verifier-side, outside the certified pipeline).

The certified pipeline releases the tap's (m, m_fp) inside the certificate. The
black-box comparator is NOT certified: after reveal, the caller runs it over the
revealed transcript + labels to get the comparator's (m, m_fp), and the separation
is the difference. This mirrors the formalism, where the comparator is a baseline
the separation is measured against, not part of the protocol.

The comparator is passed in by the caller (a mock in basic_mocked, a real Gemma
monitor in confidence) -- the engine stays model/eval-agnostic.
"""

from __future__ import annotations

from ..interfaces import Monitor, MonitorInput


def comparator_rates(comparator: Monitor, reveal: dict) -> dict[str, float]:
    """(FN, FP) rates of the black-box comparator over revealed (transcript, labels).

    The comparator sees only (x, y) -- the thought channel is never revealed, so
    the certified tap's rates cannot be recomputed here; they come from the
    certificate. That asymmetry is the point.
    """
    by_round = {t["i"]: t for t in reveal["transcript"]}
    tests = reveal["tests"]  # [(x, z)] in round order
    n = len(tests)
    fn = fp = 0
    for i, (x, z) in enumerate(tests, start=1):
        y = by_round[i]["y"]
        pred = comparator.predict(MonitorInput(x=x, answer=y, thought=None))
        if z == 1 and pred.z_hat == 0:
            fn += 1
        elif z == 0 and pred.z_hat == 1:
            fp += 1
    return {"m": fn / n, "m_fp": fp / n}


def separation(cert: dict, reveal: dict, comparator: Monitor) -> dict:
    """Certified tap rates (from the certificate) vs the offline comparator."""
    tap = {"m": cert["m"], "m_fp": cert["m_fp"]}
    black_box = comparator_rates(comparator, reveal)
    return {
        "tap_certified": tap,
        "black_box": black_box,
        "separation_fn": black_box["m"] - tap["m"],  # >0 means tap catches more
    }
