"""Mock experiment for the paper's secret-test gaming-resistance argument."""

from __future__ import annotations

from ct_core.certification import run_session
from util.dynamic_test_environment import StaticEnvironment
from util.treecut import generate_items

from .mocked_protocol_components import (
    MockAgentDriver, MockTapDriver, items_package, tap_package,
)


def _certified_m(program_items, tap, memorized, items) -> float:
    # Not a shortcut: this provisions E through the certified pipeline and returns
    # the released false-negative rate from the
    # certificate. Every cell in the table below is a full session.
    env = StaticEnvironment(items, n=len(items), seed=0)
    return run_session(
        items_package(program_items), tap_package(tap, memorized), env,
        agent_drivers=(MockAgentDriver,), tap_drivers=(MockTapDriver,))["m"]


def evaluate() -> dict[str, tuple[float, float]]:
    n = 10

    # Step 1: two disjoint item sets from the same generator. Different seeds, so the
    # secret test is freshly generated and shares no items with the public benchmark.
    public_items = generate_items(n, seed=1)
    secret_items = generate_items(n, seed=2)
    program_items = public_items + secret_items

    # Step 2: the two taps. The honest one reads the program's reasoning, so it
    # generalizes. The gamed one only memorized which public items were unsafe, which is
    # exactly what a provider overfitting to a published benchmark would produce.
    memorized = {it["x"] for it in public_items if it["z"] == 1}

    # Step 3: certify each tap twice, once on the public benchmark and once on the secret
    # test. The gamed tap looks perfect on the left and falls apart on the right, and the
    # certificate reports that true secret-test loss honestly.
    return {
        "honest": (
            _certified_m(program_items, "honest", (), public_items),
            _certified_m(program_items, "honest", (), secret_items),
        ),
        "gamed": (
            _certified_m(program_items, "gamed", memorized, public_items),
            _certified_m(program_items, "gamed", memorized, secret_items),
        ),
    }
