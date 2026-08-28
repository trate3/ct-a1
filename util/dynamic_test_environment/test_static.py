"""Tests for the shared StaticEnvironment (E_D)."""

from __future__ import annotations

from util.dynamic_test_environment import StaticEnvironment


def test_static_environment_samples_and_is_deterministic():
    items = [{"x": f"q{i}", "z": i % 2, "answer": str(i)} for i in range(10)]
    env = StaticEnvironment(items, n=6, seed=3)
    configured = env.package_config()["items"]
    assert env.trial_count() == 6
    assert all(env.new_trial(i, b"coins").next_query(()) for i in range(1, 7))
    assert {item["z"] for item in configured} <= {0, 1}
    assert StaticEnvironment(items, 6, 3).package_config() == env.package_config()
