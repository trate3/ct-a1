"""Tests for the confidence TreeCut environments (no models)."""

from __future__ import annotations

from evals.confidence.environments import (
    AdaptiveEnvironment,
    GeneratedEnvironment,
    TreeCutEnvironment,
)


def test_treecut_environment_labels_balanced_and_deterministic():
    env = TreeCutEnvironment(8, seed=7)
    assert env.trial_count() == 8
    labels = [item["z"] for item in env.items()]
    assert set(labels) <= {0, 1} and 0 in labels and 1 in labels
    assert [item["z"] for item in TreeCutEnvironment(8, seed=7).items()] == labels


def test_generated_environment_multistratum():
    env = GeneratedEnvironment(8, seed=7)
    items = env.package_config()["items"]
    assert env.trial_count() == 8
    assert all(isinstance(item["x"], str) and item["x"] for item in items)
    assert {item["z"] for item in items} <= {0, 1}


def test_adaptive_escalates_on_hallucination():
    # A program that always asserts a number gets every unanswerable item escalated,
    # concentrating unanswerable items above the 50/50 base mix.
    n = 12
    env = AdaptiveEnvironment(n, seed=7)
    for i in range(1, n + 1):
        env.next_query(i)
        env.observe(i, "42")
    labels = env.labels()
    assert len(labels) == n
    assert sum(labels) > n // 2


def test_adaptive_no_escalation_when_program_abstains():
    n = 12
    env = AdaptiveEnvironment(n, seed=7)
    for i in range(1, n + 1):
        env.next_query(i)
        env.observe(i, "unknown")
    labels = env.labels()
    assert len(labels) == n
    assert sum(labels) <= n // 2 + 1
