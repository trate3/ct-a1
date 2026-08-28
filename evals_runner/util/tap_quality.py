"""Offline tap-quality analysis over the existing direct evaluation harness."""

from __future__ import annotations

import random
from typing import Callable

from ct_core.agent import Agent
from ct_core.certification import measure_loaded, prepare_environment
from ct_core.interfaces import Monitor, MonitorInput, Prediction, TestEnvironment
from ct_core.protocol.digests import tests_digest

SCHEMA_VERSION = "ct-tap-quality/1"


class _RecordingMonitor(Monitor):
    def __init__(self, tap: Monitor, comparator: Monitor | None, recordings: dict):
        self._tap, self._comparator = tap, comparator
        self._recordings = recordings
        self.needs_activations, self.probe_layers = tap.needs_activations, tap.probe_layers

    def predict(self, inp: MonitorInput) -> Prediction:
        prediction = self._tap.predict(inp)
        self._recordings["tap"].append(prediction)
        if self._comparator:
            black_box_input = MonitorInput(x=inp.x, answer=inp.answer)
            self._recordings["comparator"].append(
                self._comparator.predict(black_box_input))
        return prediction

    def digest(self) -> str:
        return self._tap.digest()

def _bootstrap_ci(
    values: list[float], seed: int, cluster_size: int, samples: int
) -> list | None:
    if cluster_size <= 0 or len(values) % cluster_size:
        raise ValueError("prediction count must be divisible by a positive cluster_size")
    clusters = [values[i:i + cluster_size] for i in range(0, len(values), cluster_size)]
    if len(clusters) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(
        sum(sum(rng.choice(clusters)) for _ in clusters) / len(values)
        for _ in range(samples)
    )
    return [means[round((samples - 1) * 0.025)], means[round((samples - 1) * 0.975)]]


def _losses(predictions: list[Prediction], labels: list[int]) -> tuple[list, list]:
    fn = [float(label == 1 and pred.z_hat == 0) for pred, label in zip(predictions, labels)]
    fp = [float(label == 0 and pred.z_hat == 1) for pred, label in zip(predictions, labels)]
    return fn, fp


def _auc(predictions: list[Prediction], labels: list[int]) -> float | None:
    positive = [pred.score for pred, label in zip(predictions, labels) if label]
    negative = [pred.score for pred, label in zip(predictions, labels) if not label]
    if not positive or not negative:
        return None
    wins = sum(1 if p > n else 0.5 if p == n else 0 for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def _metrics(predictions, labels, seed, cluster_size, samples) -> tuple[dict, tuple]:
    if len(predictions) != len(labels) or not labels:
        raise ValueError("predictions and labels must have the same positive length")
    fn_loss, fp_loss = _losses(predictions, labels)
    positives, n = sum(labels), len(labels)
    fn, fp = int(sum(fn_loss)), int(sum(fp_loss))
    predicted_positives = sum(pred.z_hat for pred in predictions)
    metrics = {
        "n": n, "positives": positives, "negatives": n - positives,
        "predicted_positives": predicted_positives,
        "prediction_rate": predicted_positives / n,
        "mean_score": sum(pred.score for pred in predictions) / n,
        "false_negatives": fn, "false_positives": fp,
        "m": fn / n, "m_ci95": _bootstrap_ci(fn_loss, seed, cluster_size, samples),
        "m_fp": fp / n, "m_fp_ci95": _bootstrap_ci(fp_loss, seed + 1, cluster_size, samples),
        "fnr": fn / positives if positives else None,
        "fpr": fp / (n - positives) if n > positives else None,
        "roc_auc": _auc(predictions, labels),
    }
    return metrics, (fn_loss, fp_loss)


def _run_tap_quality_rounds(build_agent, build_tap, environment, *, seed,
                            build_comparator, cluster_size, bootstrap_samples):
    """Paired offline replay over a ct/0.1 round-based environment.

    One agent execution per round, scored by BOTH monitors, so the tap and the
    comparator are compared on identical outputs and the pairing is real rather than
    two independent runs differenced. This is the offline analysis only: it is not the
    certified measurement, and unlike the certified pass it may see the thought channel,
    which is exactly why it is kept outside the certificate.

    Statistics come from the same `_metrics` the ct/0.2 path uses, so the two cases
    report the same quantities computed the same way.
    """
    from ct_core.interfaces import MonitorInput

    agent, tap = build_agent(), build_tap()
    comparator = build_comparator() if build_comparator else None
    capture = tap.probe_layers if getattr(tap, "needs_activations", False) else None

    tap_predictions, comparator_predictions, queries = [], [], []
    for index in range(1, environment.n() + 1):
        x = environment.next_query(index)
        generation = agent.generate(x, capture_layers=capture)
        view = MonitorInput(x=x, answer=generation.answer, thought=generation.thought,
                            activations=generation.activations)
        tap_predictions.append(tap.predict(view))
        if comparator is not None:
            # The comparator is black-box by construction: same round, no thought.
            comparator_predictions.append(comparator.predict(
                MonitorInput(x=x, answer=generation.answer, thought=None)))
        environment.observe(index, generation.answer)
        queries.append(x)

    labels = environment.labels()
    tests = list(zip(queries, labels))
    tap_metrics, tap_losses = _metrics(
        tap_predictions, labels, seed, cluster_size, bootstrap_samples)
    result = {
        "schema_version": SCHEMA_VERSION,
        "test_digest": tests_digest(tests),
        "bootstrap": {"samples": bootstrap_samples, "seed": seed,
                      "cluster_size": cluster_size},
        "tap": tap_metrics,
    }
    if comparator is None:
        return result
    comparator_metrics, comparator_losses = _metrics(
        comparator_predictions, labels, seed, cluster_size, bootstrap_samples)
    difference = {}
    for index, name in enumerate(("m", "m_fp")):
        per_item = [b - a for a, b in zip(tap_losses[index], comparator_losses[index])]
        difference[name] = sum(per_item) / len(per_item)
        difference[f"{name}_ci95"] = _bootstrap_ci(
            per_item, seed + index + 2, cluster_size, bootstrap_samples)
    result.update(comparator=comparator_metrics, comparator_minus_tap=difference)
    return result


def _factory(value):
    """Accept either a zero-argument builder or an already-constructed object.

    The ct/0.2 cases pass builders because `measure` requires a fresh agent and tap per
    trial. The grafted ct/0.1 conflict case passes live objects instead, and reuses one
    across rounds. Supporting both keeps that case running unmodified without changing
    what the existing callers do -- a builder is still called once per trial, and a live
    object is still shared, which is each design's own intent rather than a compromise.
    """
    return value if callable(value) else (lambda: value)


def run_tap_quality(
    build_agent: "Callable[[], Agent] | Agent",
    build_tap: "Callable[[], Monitor] | Monitor",
    environment: TestEnvironment, *, seed: int,
    build_comparator: "Callable[[], Monitor] | Monitor | None" = None,
    comparator: "Monitor | None" = None,
    cluster_size: int = 1,
    bootstrap_samples: int = 2_000,
) -> dict:
    """Evaluate a tap and optional output-only comparator on identical executions.

    `comparator` is the ct/0.1 spelling of `build_comparator`; supplying both is an
    error rather than a silent precedence rule.
    """
    if comparator is not None and build_comparator is not None:
        raise TypeError("pass either comparator or build_comparator, not both")
    build_comparator = build_comparator if comparator is None else comparator
    build_agent, build_tap = _factory(build_agent), _factory(build_tap)
    if build_comparator is not None:
        build_comparator = _factory(build_comparator)
    # ct/0.1 environments (the grafted conflict case) expose n()/next_query()/observe()
    # /labels() and have no registered driver, so the ct/0.2 replay below -- which packs
    # and seals the environment and reconstructs it through `runtime_driver` -- cannot
    # drive them. Dispatch on the capability rather than on the case name.
    if not getattr(type(environment), "DRIVER", None) and hasattr(environment, "labels"):
        return _run_tap_quality_rounds(
            build_agent, build_tap, environment, seed=seed,
            build_comparator=build_comparator, cluster_size=cluster_size,
            bootstrap_samples=bootstrap_samples)

    recordings = {"tap": [], "comparator": []}

    def build_recorder():
        return _RecordingMonitor(
            build_tap(), build_comparator() if build_comparator else None,
            recordings)

    spec, packed = prepare_environment(environment)
    tests = measure_loaded(
        build_agent, build_recorder, packed, spec,
        (environment.runtime_driver(),)).pop("_tests")
    labels = [label for _, label in tests]
    tap_metrics, tap_losses = _metrics(
        recordings["tap"], labels, seed, cluster_size, bootstrap_samples)
    result = {
        "schema_version": SCHEMA_VERSION,
        "test_digest": tests_digest(tests),
        "bootstrap": {"samples": bootstrap_samples, "seed": seed,
                      "cluster_size": cluster_size},
        "tap": tap_metrics,
    }
    if not build_comparator:
        return result

    comparator_metrics, comparator_losses = _metrics(
        recordings["comparator"], labels, seed, cluster_size, bootstrap_samples)
    difference = {}
    for index, name in enumerate(("m", "m_fp")):
        per_item = [b - a for a, b in zip(tap_losses[index], comparator_losses[index])]
        difference[name] = sum(per_item) / len(per_item)
        difference[f"{name}_ci95"] = _bootstrap_ci(
            per_item, seed + index + 2, cluster_size, bootstrap_samples)
    result.update(comparator=comparator_metrics, comparator_minus_tap=difference)
    return result
