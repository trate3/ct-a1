"""Evaluation-only direct/certified comparison and JSON output."""

from __future__ import annotations

import json
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ct_core.certification import (
    SessionTelemetry, measure, prepare_components, prepare_environment, run_session,
)

from .telemetry import detect_accelerator, host_provenance


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str
    seed: int
    n_trials: int
    environment: str
    parameters: dict = field(default_factory=dict)

    def as_dict(self):
        if not self.experiment or not self.environment or self.n_trials <= 0:
            raise ValueError("experiment, environment, and positive n_trials are required")
        value = asdict(self)
        json.dumps(value)
        return value


def _timed(call, accelerator=None):
    accelerator = accelerator or detect_accelerator()
    accelerator.reset()
    before = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    value = call()
    # Device work is launched asynchronously, so the host clock must not stop until
    # the accelerator has drained. On CPU both calls are no-ops.
    accelerator.synchronize()
    elapsed = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_SELF)
    rss = after.ru_maxrss if sys.platform == "darwin" else after.ru_maxrss * 1024
    return value, {
        "wall_seconds": elapsed,
        "cpu_seconds": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        "process_lifetime_max_rss_bytes": int(rss), "token_counts": None,
        "accelerator": accelerator.peak(),
    }


def _phase_delta(direct_metrics, certified_metrics, name):
    """Certified-minus-direct for one sub-phase, or None if either arm lacks it."""
    before = direct_metrics.get("phases", {}).get(name)
    after = certified_metrics.get("phases", {}).get(name)
    if before is None or after is None:
        return None
    return after - before


def _device_peak_delta(direct_metrics, certified_metrics):
    """Certified-minus-direct device peak, or None when there is no accelerator."""
    key = "device_peak_allocated_bytes"
    direct, certified = direct_metrics["accelerator"], certified_metrics["accelerator"]
    if not direct or not certified:
        return None
    return certified[key] - direct[key]


def run_comparison(config, *, build_agent_package, build_tap_package,
                   agent_drivers, tap_drivers, build_environment, certified_first=False):
    # One probe for both arms: the peak counter is reset per arm, while the device
    # description is a property of the machine rather than of either arm.
    accelerator = detect_accelerator()

    def direct():
        environment = build_environment()
        spec, packed = prepare_environment(environment)
        if spec["n_trials"] != config.n_trials:
            raise ValueError("environment trial count differs from config")
        # Split inside the timed region, not around it: the outer wall time stays
        # exactly what it was, and the two components become separable. Committing the
        # artifacts means hashing every provisioned file, which for a multi-gigabyte
        # checkpoint dwarfs the per-trial cost and has nothing to do with it.
        phases = {}

        def body():
            started = time.perf_counter()
            components = prepare_components(
                build_agent_package(), build_tap_package(), agent_drivers, tap_drivers)
            phases["commitment_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            measured = measure(components, packed, spec, (environment.runtime_driver(),))
            phases["model_run_seconds"] = time.perf_counter() - started
            return measured

        result, metrics = _timed(body, accelerator)
        metrics["phases"] = phases
        for private in ("_tests", "_steps", "_trial_predictions", "_trial_labels"):
            result.pop(private)
        return result, metrics, spec

    def certified():
        environment = build_environment()
        traffic = SessionTelemetry()
        phases = {}
        cert, metrics = _timed(
            lambda: run_session(
                build_agent_package(), build_tap_package(), environment,
                agent_drivers=agent_drivers, tap_drivers=tap_drivers, telemetry=traffic,
                phases=phases),
            accelerator)
        metrics["phases"] = phases
        return cert, metrics, traffic

    if certified_first:
        cert, certified_metrics, traffic = certified()
        direct_result, direct_metrics, spec = direct()
    else:
        direct_result, direct_metrics, spec = direct()
        cert, certified_metrics, traffic = certified()
    if direct_result != {"m": cert["m"], "m_fp": cert["m_fp"]}:
        raise ValueError("certification changed the measurement")
    direct_metrics["transmitted_bytes_total"] = 0
    direct_metrics["transmitted_bytes_by_message"] = {}
    certified_metrics["transmitted_bytes_total"] = (
        traffic.sent_bytes + traffic.received_bytes)
    certified_metrics["transmitted_bytes_by_message"] = traffic.by_message
    # The `test` message carries E sealed and padded to a fixed length, so it is both
    # the bulk of the traffic and invariant to the workload and the trial count. Split
    # out so a reader can see the cost of running the protocol separately from the
    # cost of delivering a deliberately size-hiding environment.
    def message_bytes(kind):
        entry = traffic.by_message.get(kind, {})
        return entry.get("sent_bytes", 0) + entry.get("received_bytes", 0)

    # Named for what each message actually is. The bulk of the traffic is the secret
    # test environment travelling TO the workload, sealed and padded to a fixed length
    # -- it is not a certificate and it is not compressed. The certificate is the
    # `certout` message, and the commitment digest h_c rides inside `commit`.
    certified_metrics["transmitted_environment_bytes"] = message_bytes("test")
    certified_metrics["transmitted_commit_bytes"] = message_bytes("commit")
    certified_metrics["transmitted_certificate_bytes"] = message_bytes("certout")
    return {
        "schema_version": "ct-eval/3", "config": config.as_dict(),
        "execution_order": "certified-first" if certified_first else "direct-first",
        "runtime": host_provenance(accelerator),
        "environment_digest": spec["id_E"],
        "direct": {"measurement": direct_result, "metrics": direct_metrics},
        "certified": {"certificate": cert, "metrics": certified_metrics},
        "certified_wire_scope": "certifier-enclave-socketpair",
        "overhead": {
            "wall_seconds": certified_metrics["wall_seconds"] - direct_metrics["wall_seconds"],
            "cpu_seconds": certified_metrics["cpu_seconds"] - direct_metrics["cpu_seconds"],
            "transmitted_bytes_total": certified_metrics["transmitted_bytes_total"],
            "transmitted_environment_bytes":
                certified_metrics["transmitted_environment_bytes"],
            "transmitted_commit_bytes": certified_metrics["transmitted_commit_bytes"],
            "transmitted_certificate_bytes":
                certified_metrics["transmitted_certificate_bytes"],
            # The comparison's own arms both commit the artifacts, so that cost
            # cancels out of the delta above and is invisible in it -- even though it
            # is a real and often dominant cost of certifying at all. Carried here so
            # a report can state it instead of losing it.
            "commitment_seconds": _phase_delta(
                direct_metrics, certified_metrics, "commitment_seconds"),
            "model_run_seconds": _phase_delta(
                direct_metrics, certified_metrics, "model_run_seconds"),
            "device_peak_allocated_bytes": _device_peak_delta(
                direct_metrics, certified_metrics),
        },
    }


def write_result(result: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return destination


def write_certificate(cert: dict, spec: dict, output: str) -> dict:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "cert.json").write_text(json.dumps(cert, indent=2) + "\n")
    (destination / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    return cert
