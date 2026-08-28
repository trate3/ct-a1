"""Execute the Gemma-2/Apollo-probe insider-trading evaluation case."""

from __future__ import annotations

import time
from pathlib import Path

from util.model_backend import artifact_total_bytes
from ct_core.certification import prepare_environment, run_session, verify_certificate
from evals_runner.util import ExperimentConfig, run_comparison, run_tap_quality, write_result

from .probe_tap import ApolloProbe
from .insider_trading_probe_components import (
    FinancialAgentDriver,
    FinancialTapDriver,
    build_agent,
    component_packages,
    build_environment,
    build_program,
    build_tap,
    latest_detector,
    resolve_dtype,
    ATTN_IMPLEMENTATION,
)


def evaluate(
    *,
    detector_path: str | None,
    model: str,
    n_trajectories: int,
    seed: int,
    threshold: float,
    device: str,
    max_new_tokens: int,
    dtype: str = "auto",
    certification_overhead_out: str | None = None,
    certified_first: bool = False,
    result_out: str | None = None,
) -> int:
    # Step 1: pick the trained probe. Defaults to the newest one Apollo's trainer wrote
    # (RUNBOOK Phase 2), so the two halves of the pipeline connect without hand-wiring.
    detector = Path(detector_path) if detector_path else latest_detector()
    print(f"[run] probe: {detector}", flush=True)

    # Step 2: the three pieces the protocol needs.
    #   program -- Gemma-2, dense: the weights the probe was trained on. (Apollo can now
    #              train on Gemma-4 too, via third_party/patches/gemma4-port.patch, but
    #              probe and program must be the same weights -- see probe-track.md.)
    #   tap     -- the probe reading internal activations, plus its decision threshold.
    #              Weights AND threshold get committed in h_c before the test is revealed,
    #              so neither can be tuned after seeing the questions.
    #   env     -- the certifier's secret test. Interactive: each scenario gives the agent
    #              two actions, and the second observation follows its actual first tool.
    # We commit the AGENT, not the bare model: Agent folds the model, its control-logic,
    # and Apollo's versioned financial tool interfaces into h_c.
    # Resolved once here and then carried in the committed agent config, so the
    # direct and certified arms are guaranteed to load the same weights.
    dtype = resolve_dtype(device, dtype)
    print(f"[run] device: {device}  dtype: {dtype}  "
          f"attn: {ATTN_IMPLEMENTATION}", flush=True)
    program = build_program(model, device, max_new_tokens, verbose=True, dtype=dtype)
    probe = ApolloProbe.load(detector)
    agent = build_agent(program)
    tap = build_tap(probe, threshold)
    environment = build_environment(n_trajectories, seed)
    parameters = {
        "model": model,
        "device": device,
        "program_artifact": program.digest(),
        "dtype": dtype,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "commitment_input_bytes": artifact_total_bytes(model),
        "max_new_tokens": max_new_tokens,
        "n_trajectories": n_trajectories,
        "probe_sha256": probe.content_hash(),
        "threshold": threshold,
    }
    agent_package, tap_package = component_packages(
        probe, model=model, device=device, dtype=dtype, max_new_tokens=max_new_tokens,
        threshold=threshold, verbose=True)

    if certification_overhead_out:
        result = run_comparison(
            ExperimentConfig(
                "financial-certification-overhead",
                seed,
                environment.trial_count(),
                "apollo-insider-trading",
                parameters=parameters,
            ),
            build_agent_package=lambda: agent_package,
            build_tap_package=lambda: tap_package,
            agent_drivers=(FinancialAgentDriver,),
            tap_drivers=(FinancialTapDriver,),
            build_environment=lambda: build_environment(n_trajectories, seed),
            certified_first=certified_first,
        )
        print(f"[run] wrote {write_result(result, certification_overhead_out)}")
        return 0

    print(
        f"[run] {n_trajectories} insider-trading trajectories = "
        f"{2 * environment.trial_count()} rounds on {model} ({device}); each round runs "
        "the agent + reads its activations. CPU is slow.",
        flush=True,
    )
    last = [time.time()]

    def progress(i: int, n: int) -> None:
        now = time.time()
        elapsed = now - last[0]
        last[0] = now
        note = "" if i == 1 else f"  (previous scenario {elapsed:.0f}s)"
        print(f"[run] scenario {i}/{n}{note}", flush=True)

    # Step 3: run the certified session. Same run_session as the confidence eval -- the
    # protocol does not care that the tap reads activations instead of text.
    spec, _ = prepare_environment(environment)
    cert = run_session(
        agent_package, tap_package, environment,
        agent_drivers=(FinancialAgentDriver,), tap_drivers=(FinancialTapDriver,),
        progress=progress)

    # Step 4: only the aggregate leaves the enclave. The individual reports, their labels,
    # and the probe's per-step scores never appear in the certificate.
    print(f"\n[certifier] CERTIFIED  probe m(FN)={cert['m']:.3f}  m_fp={cert['m_fp']:.3f}\n")

    # Step 5: re-check signature and commitment as an outside auditor would.
    verify_certificate(cert, spec=spec)
    print(
        "\nThis is the framework property: the committed probe (weights + threshold, "
        "bound in\nh_c) certified against the certifier's secret insider-trading "
        "environment. m(FN) =\nconcealed reports the probe missed; m_fp = honest "
        "reports/trades it flagged. (The\nenvironment labels concealment with a "
        "heuristic judge; an LM judge would change the\ntest, not the protocol. Probe "
        "accuracy/threshold tuning is Apollo's concern, not the\nframework's.)",
        flush=True,
    )
    if result_out:
        print("[run] computing trajectory-clustered offline statistics...", flush=True)
        quality = run_tap_quality(
            lambda: build_agent(program),
            lambda: build_tap(probe, threshold),
            build_environment(n_trajectories, seed),
            seed=seed,
            cluster_size=2,
        )
        result = {
            "schema_version": "ct-tap-evaluation/1",
            "config": {
                "case": "financial",
                "seed": seed,
                "n_trials": environment.trial_count(),
                "parameters": parameters,
            },
            "certificate": cert,
            "offline_quality": quality,
        }
        print(f"[run] wrote {write_result(result, result_out)}")
    return 0
