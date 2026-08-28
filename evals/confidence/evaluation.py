"""Execute the real Gemma-4 CoT-confidence evaluation case."""

from __future__ import annotations

import time

from util.model_backend import artifact_total_bytes
from ct_core.certification import prepare_environment, run_session, verify_certificate
from evals_runner.util import ExperimentConfig, run_comparison, run_tap_quality, write_result

from .cot_confidence_components import (
    ConfidenceAgentDriver,
    ConfidenceTapDriver,
    build_agent,
    build_backends,
    build_comparator,
    build_environment,
    build_tap,
    component_packages,
)


def _progress():
    last = [time.time()]

    def report(i: int, n: int) -> None:
        now = time.time()
        elapsed = now - last[0]
        last[0] = now
        note = "" if i == 1 else f"  (previous item {elapsed:.0f}s)"
        print(f"[run] round {i}/{n}{note}", flush=True)

    return report


def evaluate(
    *,
    program_model: str,
    monitor_model: str | None,
    n: int,
    seed: int,
    device: str,
    dtype: str,
    max_new_tokens: int,
    certification_overhead_out: str | None = None,
    certified_first: bool = False,
    result_out: str | None = None,
) -> int:
    # Step 1: load the weights. One backend if the tap runs on the same model as the
    # program, two if --monitor-model asks for the faithful setup (big program, small
    # monitor reading its reasoning).
    program_backend, monitor_backend = build_backends(
        program_model, monitor_model, device, dtype
    )

    # Step 2: the controlled comparison. The tap and comparator use the same model and
    # seeded trials; the only difference is use_thought, i.e. whether it gets to read
    # the program's chain of thought or only its final answer. Any separation below is
    # therefore attributable to thought-channel access and nothing else.
    agent = build_agent(program_backend, max_new_tokens, verbose=True)
    tap = build_tap(monitor_backend, verbose=True)
    parameters = {
        "program_model": program_model,
        "monitor_model": monitor_model or program_model,
        "device": device,
        "program_artifact": program_backend.digest(),
        "monitor_artifact": monitor_backend.digest(),
        "dtype": dtype,
        "commitment_input_bytes": artifact_total_bytes(program_model)
        + (artifact_total_bytes(monitor_model)
           if monitor_model and monitor_model != program_model else 0),
        "max_new_tokens": max_new_tokens,
    }

    if certification_overhead_out:
        result = run_comparison(
            ExperimentConfig(
                "confidence-certification-overhead",
                seed,
                n,
                "treecut-generated",
                parameters=parameters,
            ),
            build_agent_package=lambda: component_packages(
                program_model, monitor_model, device, dtype, max_new_tokens,
                verbose=True)[0],
            build_tap_package=lambda: component_packages(
                program_model, monitor_model, device, dtype, max_new_tokens,
                verbose=True)[1],
            agent_drivers=(ConfidenceAgentDriver,),
            tap_drivers=(ConfidenceTapDriver,),
            build_environment=lambda: build_environment(n, seed),
            certified_first=certified_first,
        )
        print(f"[run] wrote {write_result(result, certification_overhead_out)}")
        return 0

    # Each round runs a real generation (up to --max-new-tokens) plus a judge pass, so
    # on CPU/MPS a round is seconds to minutes. The program and tap log their own
    # sub-steps; this prints a round header with the previous item's wall time so a
    # long run visibly advances instead of looking hung.
    print(
        f"[run] {n} items, up to {max_new_tokens} new tokens each; "
        "this can take a while on CPU/MPS.",
        flush=True,
    )

    # Step 3: C supplies E once; the enclave executes it and releases only aggregates.
    environment = build_environment(n, seed)
    spec, _ = prepare_environment(environment)
    agent_package, tap_package = component_packages(
        program_model, monitor_model, device, dtype, max_new_tokens, verbose=True)
    cert = run_session(
        agent_package, tap_package, environment,
        agent_drivers=(ConfidenceAgentDriver,), tap_drivers=(ConfidenceTapDriver,),
        progress=_progress())
    print(f"\n[certifier] CERTIFIED  m(FN)={cert['m']:.3f}  m_fp={cert['m_fp']:.3f}\n")

    # Step 4: re-check signature and public specification binding as a relying party.
    verify_certificate(cert, spec=spec)

    # Step 5: measure quality separately; certification does not reveal its trials.
    print(f"[run] computing offline black-box comparison over {n} items...", flush=True)
    quality = run_tap_quality(
        lambda: build_agent(program_backend, max_new_tokens, verbose=True),
        lambda: build_tap(monitor_backend, verbose=True),
        build_environment(n, seed),
        seed=seed,
        build_comparator=lambda: build_comparator(monitor_backend, verbose=True),
    )
    comparison = {
        "tap_certified": {"m": cert["m"], "m_fp": cert["m_fp"]},
        "black_box": quality["comparator"],
        "separation_fn": quality["comparator_minus_tap"]["m"],
        "separation_fp": quality["comparator_minus_tap"]["m_fp"],
    }
    print("\n--- separation (offline, tap vs black-box) ---")
    print(
        f"  tap (certified) m(FN)={comparison['tap_certified']['m']:.3f}  "
        f"black-box m(FN)={comparison['black_box']['m']:.3f}  "
        f"separation={comparison['separation_fn']:+.3f}"
    )
    if result_out:
        result = {
            "schema_version": "ct-tap-evaluation/1",
            "config": {"case": "confidence", "seed": seed, "n_trials": n,
                       "parameters": parameters},
            "certificate": cert,
            "certified_comparison": comparison,
            "offline_quality": quality,
        }
        print(f"[run] wrote {write_result(result, result_out)}")
    return 0
