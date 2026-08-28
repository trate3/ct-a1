"""Uncertified reference execution for framework-overhead comparisons.

This path runs the same committed ``Agent``, ``Monitor``, and ``TestEnvironment``
interfaces without commitments, encryption, signatures, or sockets.  It is not a
security mode and produces no certificate; it supplies the direct-computation
baseline for evaluation.
"""

from __future__ import annotations

from ..agent import Agent
from ..interfaces import Monitor, MonitorInput, TestEnvironment


def run_direct(agent: Agent, tap: Monitor, env: TestEnvironment) -> dict:
    """Run one direct measurement and return aggregates plus its labeled tests."""
    n = env.n()
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("environment round count must be a positive integer")

    capture = tap.probe_layers if getattr(tap, "needs_activations", False) else None
    xs: list[str] = []
    zhats: list[int] = []
    for i in range(1, n + 1):
        x = env.next_query(i)
        if not isinstance(x, str):
            raise ValueError(f"round {i}: query must be a string")
        gen = agent.generate(x, capture_layers=capture)
        pred = tap.predict(MonitorInput(
            x=x, answer=gen.answer, thought=gen.thought, activations=gen.activations))
        if pred.z_hat not in (0, 1):
            raise ValueError(f"round {i}: tap prediction must be binary")
        env.observe(i, gen.answer)
        xs.append(x)
        zhats.append(pred.z_hat)

    labels = env.labels()
    if (not isinstance(labels, list) or len(labels) != n
            or any(z not in (0, 1) for z in labels)):
        raise ValueError("environment labels must be one binary value per round")
    m = sum(1 for zh, z in zip(zhats, labels) if z == 1 and zh == 0) / n
    m_fp = sum(1 for zh, z in zip(zhats, labels) if z == 0 and zh == 1) / n
    return {"m": m, "m_fp": m_fp, "_tests": list(zip(xs, labels))}
