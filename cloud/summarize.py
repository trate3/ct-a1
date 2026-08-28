#!/usr/bin/env python3
"""Reduce the two counterbalanced arms to one comparable record.

Usage: summarize.py <run-directory>   (expects direct_first.json, certified_first.json)

What this does and does not claim:

  * The certification overhead reported here is the *counterbalanced* estimate --
    the mean of (certified - direct) taken separately within each process, so a
    once-per-process cost (CUDA context, allocator warm-up, cuDNN autotuning) lands
    in both arms of a pair and cancels instead of being attributed to the protocol.
  * `wire_bytes` is the traffic on one in-process socketpair between the certifier
    and the trusted workload. It is an exact byte count for that hop and is not a
    network measurement.
  * Host RSS is only meaningful for the FIRST arm of each process, because
    ru_maxrss is a process-lifetime high-water mark that never resets. On the GPU
    the weights are in device memory anyway; read the accelerator peaks instead.
  * Nothing here is a TEE measurement. The workload and the certifier are separate
    protocol endpoints inside one process on a general-purpose VM.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ARMS = ("direct_first", "certified_first")


def gib(value):
    return None if value is None else value / 1024**3


def load(directory: Path) -> dict:
    runs = {}
    for arm in ARMS:
        path = directory / f"{arm}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        runs[arm] = json.loads(path.read_text())
    return runs


def check_identical(runs: dict) -> dict:
    """The two orderings must be the same experiment, or the pair is not comparable."""
    problems = []
    reference = runs[ARMS[0]]
    # h_c is deliberately absent: it is a *hiding* commitment, salted with a fresh
    # secrets.token_bytes randomizer per session, so two honest runs of the identical
    # experiment must produce different h_c. Comparing it would flag correct behavior
    # as a mismatch. The deterministic bindings are h_s, id_E, and the config -- which
    # is what actually has to match for the two orderings to be one paired measurement.
    fields = {
        "environment_digest": lambda r: r["environment_digest"],
        "specification h_s": lambda r: r["certified"]["certificate"]["h_s"],
        "released (m, m_fp)": lambda r: (r["certified"]["certificate"]["m"],
                                        r["certified"]["certificate"]["m_fp"]),
        "config": lambda r: r["config"],
    }
    for label, get in fields.items():
        values = {arm: get(runs[arm]) for arm in ARMS}
        if values[ARMS[0]] != values[ARMS[1]]:
            problems.append(f"{label} differs between arms: {values}")
    for arm, run in runs.items():
        direct = run["direct"]["measurement"]
        cert = run["certified"]["certificate"]
        if direct != {"m": cert["m"], "m_fp": cert["m_fp"]}:
            problems.append(f"{arm}: direct and certified measurements disagree")
    return {
        "identical_experiment": not problems,
        "problems": problems,
        "environment_digest": reference["environment_digest"],
        "specification_h_s": reference["certified"]["certificate"]["h_s"],
        "commitment_h_c_per_arm": {
            arm: runs[arm]["certified"]["certificate"]["h_c"] for arm in ARMS},
        "released": {"m": reference["certified"]["certificate"]["m"],
                     "m_fp": reference["certified"]["certificate"]["m_fp"]},
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    directory = Path(sys.argv[1])
    runs = load(directory)
    identity = check_identical(runs)

    runtime = runs[ARMS[0]]["runtime"]
    accelerator = runtime.get("accelerator") or {}
    instance = runtime.get("instance") or {}
    config = runs[ARMS[0]]["config"]

    print(f"\nrun            : {directory}")
    print(f"experiment     : {config['experiment']}")
    print(f"environment    : {config['environment']}  "
          f"n_trials={config['n_trials']}  seed={config['seed']}")
    parameters = config.get("parameters", {})
    print(f"model / dtype  : {parameters.get('model')}  "
          f"{parameters.get('device')}/{parameters.get('dtype')}  "
          f"max_new_tokens={parameters.get('max_new_tokens')}")
    print(f"probe          : sha256 {str(parameters.get('probe_sha256'))[:16]}...  "
          f"threshold={parameters.get('threshold')}")
    total = gib(accelerator.get("total_memory_bytes"))
    print(f"accelerator    : {accelerator.get('name')}  "
          f"{'-' if total is None else format(total, '.0f') + ' GiB'}  "
          f"driver {accelerator.get('driver')}  cuda {accelerator.get('cuda_runtime')}")
    print(f"instance       : {instance.get('instance_name')} "
          f"{instance.get('machine_type')} {instance.get('zone')} "
          f"({instance.get('provisioning_model')})")
    print(f"versions       : torch {runtime['versions'].get('torch')}  "
          f"transformers {runtime['versions'].get('transformers')}")

    print("\nidentity check")
    print(f"  same experiment across arms : {identity['identical_experiment']}")
    for problem in identity["problems"]:
        print(f"  !! {problem}")
    print(f"  environment digest id_E     : {identity['environment_digest'][:16]}...")
    print(f"  specification h_s           : {identity['specification_h_s'][:16]}...")
    for arm, value in identity["commitment_h_c_per_arm"].items():
        print(f"  commitment h_c ({arm:<15}): {value[:16]}...  "
              f"(salted per session; differing is correct)")
    print(f"  released m(FN) / m_fp       : {identity['released']['m']:.4f} / "
          f"{identity['released']['m_fp']:.4f}")

    header = f"\n{'arm':<16} {'phase':<10} {'wall s':>10} {'cpu s':>10} " \
             f"{'GPU peak GiB':>13} {'host RSS GiB':>13} {'wire bytes':>12}"
    print(header)
    print("-" * len(header))
    per_arm = {}
    for arm in ARMS:
        run = runs[arm]
        for phase in ("direct", "certified"):
            metrics = run[phase]["metrics"]
            device = metrics.get("accelerator") or {}
            peak = gib(device.get("device_peak_allocated_bytes"))
            print(f"{arm:<16} {phase:<10} {metrics['wall_seconds']:>10.2f} "
                  f"{metrics['cpu_seconds']:>10.2f} "
                  f"{(f'{peak:.3f}' if peak is not None else '-'):>13} "
                  f"{gib(metrics['process_lifetime_max_rss_bytes']):>13.3f} "
                  f"{metrics['wire_bytes']:>12,}")
        per_arm[arm] = run["overhead"]

    print("\nper-process overhead (certified - direct)")
    for arm in ARMS:
        overhead = per_arm[arm]
        device_delta = overhead.get("device_peak_allocated_bytes")
        print(f"  {arm:<16} wall {overhead['wall_seconds']:+8.2f}s  "
              f"cpu {overhead['cpu_seconds']:+8.2f}s  "
              f"device peak {('-' if device_delta is None else f'{device_delta:+,}')} B")

    wall = [per_arm[arm]["wall_seconds"] for arm in ARMS]
    cpu = [per_arm[arm]["cpu_seconds"] for arm in ARMS]
    mean = lambda xs: sum(xs) / len(xs)
    trials = config["n_trials"]
    counterbalanced = {
        "wall_seconds": mean(wall),
        "cpu_seconds": mean(cpu),
        "wall_seconds_per_trial": mean(wall) / trials,
        "wire_bytes": mean([runs[arm]["certified"]["metrics"]["wire_bytes"]
                            for arm in ARMS]),
        "order_spread_wall_seconds": abs(wall[0] - wall[1]),
    }
    baseline = mean([runs[arm]["direct"]["metrics"]["wall_seconds"] for arm in ARMS])
    counterbalanced["wall_fraction_of_direct"] = (
        counterbalanced["wall_seconds"] / baseline if baseline else None)

    print("\ncounterbalanced certification overhead (mean over both orderings)")
    print(f"  wall            : {counterbalanced['wall_seconds']:+.2f} s "
          f"over {baseline:.2f} s direct "
          f"({100 * counterbalanced['wall_fraction_of_direct']:+.2f}%)")
    print(f"  wall per trial  : {counterbalanced['wall_seconds_per_trial']:+.3f} s "
          f"across {trials} trials")
    print(f"  cpu             : {counterbalanced['cpu_seconds']:+.2f} s")
    print(f"  wire            : {counterbalanced['wire_bytes']:,.0f} bytes "
          f"on the certifier-workload socketpair")
    print(f"  ordering spread : {counterbalanced['order_spread_wall_seconds']:.2f} s "
          f"(how much the two orderings disagree; large means one process warmed up)")

    print("\nfirst-arm host RSS (the only comparable RSS figures; ru_maxrss never resets)")
    print(f"  direct_first    / direct    "
          f"{gib(runs['direct_first']['direct']['metrics']['process_lifetime_max_rss_bytes']):.3f} GiB")
    print(f"  certified_first / certified "
          f"{gib(runs['certified_first']['certified']['metrics']['process_lifetime_max_rss_bytes']):.3f} GiB")

    summary = {
        "schema_version": "ct-cloud-overhead/1",
        "run_directory": str(directory),
        "config": config,
        "runtime": runtime,
        "identity": identity,
        "per_process_overhead": per_arm,
        "counterbalanced_overhead": counterbalanced,
        "direct_wall_seconds_mean": baseline,
        "measurement_scope": (
            "Certifier and trusted workload are separate protocol endpoints in one "
            "process on a general-purpose GCE instance. wire_bytes is the exact byte "
            "count on that in-process socketpair, not a network measurement, and none "
            "of this is a TEE measurement."),
    }
    destination = directory / "summary.json"
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {destination}")
    return 0 if identity["identical_experiment"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
