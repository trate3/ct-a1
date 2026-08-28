"""Run and record a direct-versus-certified evaluation."""

from __future__ import annotations

import json
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from ..agent import Agent
from ..interfaces import Monitor, TestEnvironment
from ..protocol.digests import tests_digest
from .direct import run_direct
from .session import SessionTelemetry, run_session

SCHEMA_VERSION = "ct-eval/1"


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str
    seed: int
    n_rounds: int
    environment: str
    parameters: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        if not self.experiment or not self.environment or self.n_rounds <= 0:
            raise ValueError("experiment, environment, and a positive n_rounds are required")
        value = asdict(self)
        json.dumps(value)  # fail before the run if the config cannot be recorded
        return value


def _measure(call: Callable[[], dict]) -> tuple[dict, dict]:
    before = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    value = call()
    after = resource.getrusage(resource.RUSAGE_SELF)
    rss = after.ru_maxrss if sys.platform == "darwin" else after.ru_maxrss * 1024
    return value, {
        "wall_seconds": time.perf_counter() - start,
        "cpu_seconds": ((after.ru_utime - before.ru_utime)
                        + (after.ru_stime - before.ru_stime)),
        "process_max_rss_bytes": int(rss),
        "token_counts": None,
    }


def run_comparison(
    config: ExperimentConfig,
    *,
    build_agent: Callable[[], Agent],
    build_tap: Callable[[], Monitor],
    build_environment: Callable[[], TestEnvironment],
) -> dict:
    """Use fresh stacks for both modes and require equivalent measurements."""
    config_dict = config.as_dict()
    direct_env = build_environment()
    if direct_env.n() != config.n_rounds:
        raise ValueError("environment round count differs from config")
    direct_agent, direct_tap = build_agent(), build_tap()
    direct, direct_metrics = _measure(lambda: run_direct(direct_agent, direct_tap, direct_env))
    direct_tests = direct.pop("_tests")
    del direct_agent, direct_tap, direct_env

    certified_env = build_environment()
    if certified_env.n() != config.n_rounds:
        raise ValueError("environment round count differs from config")
    traffic = SessionTelemetry()
    certified_agent, certified_tap = build_agent(), build_tap()
    certified, certified_metrics = _measure(
        lambda: run_session(certified_agent, certified_tap, certified_env, telemetry=traffic))

    certified.pop("_reveal")
    direct_digest = tests_digest(direct_tests)
    if direct_digest != certified["h_T"]:
        raise ValueError("direct and certified runs used different labeled tests")
    if direct != {"m": certified["m"], "m_fp": certified["m_fp"]}:
        raise ValueError("certification changed the measurement")

    direct_metrics["wire_bytes"] = 0
    certified_metrics["wire_bytes"] = traffic.sent_bytes + traffic.received_bytes
    return {
        "schema_version": SCHEMA_VERSION,
        "config": config_dict,
        "test_digest": direct_digest,
        "direct": {"measurement": direct, "metrics": direct_metrics},
        "certified": {"certificate": certified, "metrics": certified_metrics},
        "certified_wire_scope": "verifier-enclave-loopback",
        "overhead": {
            "wall_seconds": certified_metrics["wall_seconds"] - direct_metrics["wall_seconds"],
            "cpu_seconds": certified_metrics["cpu_seconds"] - direct_metrics["cpu_seconds"],
            "wire_bytes": certified_metrics["wire_bytes"],
        },
    }


def write_result(result: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return destination
