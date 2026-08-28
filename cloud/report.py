#!/usr/bin/env python3
"""Reduce repeated direct/certified runs to a cost-of-certification report.

Usage:
    report.py <run-directory> [<run-directory> ...]

Each directory holds `{direct_first,certified_first}_r<N>.json` from
cloud/run_overhead.sh. One directory per workload; pass several to get one report
covering all of them.

Statistics, and why each is the one reported:

  wall / CPU   Median across repeats, per arm. Median rather than mean because a
               single slow process (a page-cache miss, a noisy neighbour on a shared
               host) skews a mean of five far more than it moves a median, and the
               quantity of interest is the typical cost, not the worst case. Both
               orderings are pooled, so a once-per-process warm-up contributes to
               the direct and certified medians alike instead of being charged to
               the protocol.

  peak RSS     Host RSS is a process-lifetime high-water mark that never resets, so
               within one process the second arm inherits the first arm's peak and
               the two are not separable. Only the FIRST arm of each process is
               usable: direct_first's direct arm against certified_first's certified
               arm. Under GPU execution the weights live in device memory and are
               absent from this number entirely -- read the device peak instead.

  device peak  CUDA peak allocated per arm, counter reset at arm start. This is the
               memory figure that means anything once the model is on the GPU.

  wire         Bytes on the certifier-workload socketpair. Exact for that hop; not a
               network measurement. Expect it to be nearly constant across workloads
               and test-set sizes -- the environment is transmitted padded to a fixed
               length precisely so its size leaks nothing -- so this is a property of
               the protocol, not of the workload.
"""

from __future__ import annotations

import json
import re
import os
import statistics
import sys
from pathlib import Path

ARMS = ("direct_first", "certified_first")
# Cost of the machine the measurement ran on. Passed in rather than looked up here so
# a report stays reproducible offline; run_overhead.sh supplies the live catalog rate.
USD_PER_HOUR = float(os.environ.get("CT_USD_PER_HOUR", "0") or 0)
PHASES = ("direct", "certified")
KIB = 1024
GIB = 1024**3


def load_runs(directory: Path) -> list[dict]:
    runs = []
    for path in sorted(directory.glob("*_r*.json")):
        match = re.fullmatch(r"(direct_first|certified_first)_r(\d+)", path.stem)
        if not match:
            continue
        run = json.loads(path.read_text())
        run["_arm"], run["_repeat"] = match.group(1), int(match.group(2))
        runs.append(run)
    if not runs:
        raise SystemExit(f"no {{direct,certified}}_first_r<N>.json under {directory}")
    return runs


# ct-eval/2 -> ct-eval/3 renames. Older result files are still readable.
ALIASES = {
    "commitment_seconds": "commit_artifacts_seconds",
    "model_run_seconds": "evaluate_seconds",
    "transmitted_bytes_total": "wire_bytes",
    "transmitted_environment_bytes": "wire_environment_bytes",
    "commitment_input_bytes": "artifact_bytes",
}


def field(container: dict, name: str, default=None):
    """Read a metric by its current name, falling back to the pre-rename one."""
    if name in container:
        return container[name]
    return container.get(ALIASES.get(name, name), default)


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def percent(before, after):
    if not before:
        return None
    return 100.0 * (after - before) / before


def workload(directory: Path) -> dict:
    runs = load_runs(directory)
    reference = runs[0]
    config = reference["config"]

    # Every run in a directory must be the same experiment, or a median over them is
    # a median over different things.
    digests = {run["environment_digest"] for run in runs}
    configs = {json.dumps(run["config"], sort_keys=True) for run in runs}
    disagreements = []
    if len(digests) != 1:
        disagreements.append(f"{len(digests)} distinct environment digests")
    if len(configs) != 1:
        disagreements.append(f"{len(configs)} distinct configs")
    for run in runs:
        cert = run["certified"]["certificate"]
        if run["direct"]["measurement"] != {"m": cert["m"], "m_fp": cert["m_fp"]}:
            disagreements.append(f"{run['_arm']}_r{run['_repeat']}: arms disagree")

    timings = {
        phase: {
            "wall_seconds": median([r[phase]["metrics"]["wall_seconds"] for r in runs]),
            "cpu_seconds": median([r[phase]["metrics"]["cpu_seconds"] for r in runs]),
            "device_peak_bytes": median([
                (r[phase]["metrics"].get("accelerator") or {}).get(
                    "device_peak_allocated_bytes") for r in runs]),
            # Sub-phases. commit_artifacts is the cost of hashing every provisioned
            # file; evaluate is the trials themselves. Both arms pay the former, so it
            # cancels out of the direct-vs-certified delta while remaining a real and
            # often dominant cost of certifying at all.
            "commitment_seconds": median([
                field(r[phase]["metrics"].get("phases", {}), "commitment_seconds")
                for r in runs]),
            "model_run_seconds": median([
                field(r[phase]["metrics"].get("phases", {}), "model_run_seconds")
                for r in runs]),
        }
        for phase in PHASES
    }
    # Residual per process, then median -- not median(wall) - median(commit) -
    # median(evaluate), which mixes processes and inflates the spread.
    for phase in PHASES:
        residuals = []
        for r in runs:
            m = r[phase]["metrics"]
            p = m.get("phases") or {}
            if field(p, "commitment_seconds") is None:
                continue
            residuals.append(m["wall_seconds"] - field(p, "commitment_seconds")
                             - field(p, "model_run_seconds"))
        timings[phase]["other_seconds"] = median(residuals) if residuals else None
        timings[phase]["other_stdev_seconds"] = (
            statistics.stdev(residuals) if len(residuals) > 1 else None)

    # First arm of each process only -- see the module docstring.
    first_arm_rss = {
        "direct": median([r["direct"]["metrics"]["process_lifetime_max_rss_bytes"]
                          for r in runs if r["_arm"] == "direct_first"]),
        "certified": median([r["certified"]["metrics"]["process_lifetime_max_rss_bytes"]
                             for r in runs if r["_arm"] == "certified_first"]),
    }

    wire = median([field(r["certified"]["metrics"], "transmitted_bytes_total") for r in runs])
    wire_environment = median([
        field(r["certified"]["metrics"], "transmitted_environment_bytes") for r in runs])
    wire_control = median([
        (field(r["certified"]["metrics"], "transmitted_bytes_total")
         - field(r["certified"]["metrics"], "transmitted_environment_bytes", 0)) for r in runs])
    released = reference["certified"]["certificate"]
    return {
        "directory": str(directory),
        "case": config["experiment"],
        "environment": config["environment"],
        "n_trials": config["n_trials"],
        "seed": config["seed"],
        "parameters": config.get("parameters", {}),
        "commitment_input_bytes": field(
            config.get("parameters", {}), "commitment_input_bytes"),
        "runtime": reference["runtime"],
        "repeats": len({r["_repeat"] for r in runs}),
        "processes": len(runs),
        "consistent": not disagreements,
        "disagreements": disagreements,
        "released": {"m": released["m"], "m_fp": released["m_fp"]},
        "median": timings,
        "first_arm_peak_rss_bytes": first_arm_rss,
        # The protocol's own cost, read off the certified arm rather than obtained by
        # subtracting two ~270 s arms to find a sub-second quantity. The direct arm's
        # residual is the harness's zero point and should be ~0.
        "protocol_residual_seconds": (
            None if timings["certified"]["other_seconds"] is None
            else timings["certified"]["other_seconds"] - timings["direct"]["other_seconds"]),
        # Wall-clock seconds are what the instance is billed for, so every phase has
        # a price. Reported per process (both arms) and for the whole campaign.
        "usd_per_hour": USD_PER_HOUR or None,
        "campaign_wall_seconds": sum(
            r[phase]["metrics"]["wall_seconds"] for r in runs for phase in PHASES),
        "median_transmitted_bytes": wire,
        "median_transmitted_environment_bytes": wire_environment,
        "median_transmitted_control_bytes": wire_control,
        "delta": {
            "wall_seconds": timings["certified"]["wall_seconds"]
            - timings["direct"]["wall_seconds"],
            "wall_percent": percent(timings["direct"]["wall_seconds"],
                                    timings["certified"]["wall_seconds"]),
            "cpu_seconds": timings["certified"]["cpu_seconds"]
            - timings["direct"]["cpu_seconds"],
            "cpu_percent": percent(timings["direct"]["cpu_seconds"],
                                   timings["certified"]["cpu_seconds"]),
            "peak_rss_bytes": (
                None if None in first_arm_rss.values()
                else first_arm_rss["certified"] - first_arm_rss["direct"]),
            "device_peak_bytes": (
                None if timings["direct"]["device_peak_bytes"] is None
                else timings["certified"]["device_peak_bytes"]
                - timings["direct"]["device_peak_bytes"]),
        },
    }


def _wire_cell(entry: dict) -> str:
    """Total, split into control and environment when the run recorded the split."""
    total = f"{entry['median_transmitted_bytes'] / KIB:.2f} KiB"
    control = entry["median_transmitted_control_bytes"]
    environment = entry["median_transmitted_environment_bytes"]
    if control is None or environment is None:
        return total
    return (f"{total} ({control / KIB:.2f} control "
            f"+ {environment / KIB:.2f} env)")


def label(entry: dict) -> str:
    """Workload name plus trial count -- two measurement sets of the same workload at
    different trial counts are different rows and must not read identically."""
    case = entry["case"].replace("-certification-overhead", "").replace("-", " ")
    return f"{case} n={entry['n_trials']}"


def prose(entries: list[dict]) -> str:
    """The report paragraph, in the same shape as the local write-up."""
    magnitudes = []
    for entry in entries:
        pct = entry["delta"]["wall_percent"] or 0.0
        magnitudes.append("little" if abs(pct) < 2 else
                          "moderate" if abs(pct) < 15 else "substantial")
    lead = (f"At the tested workload sizes on a single NVIDIA L4, certification added "
            + " and ".join(
                f"{magnitude} overhead to the {label(entry)} workload"
                for magnitude, entry in zip(magnitudes, entries)) + ".")
    parts = [lead]
    for entry in entries:
        delta, med = entry["delta"], entry["median"]
        parts.append(
            f"For {label(entry)} evaluation, median wall time "
            f"{'increased' if delta['wall_seconds'] >= 0 else 'decreased'} from "
            f"{med['direct']['wall_seconds']:.2f} s to "
            f"{med['certified']['wall_seconds']:.2f} s "
            f"({delta['wall_percent']:+.1f}%), while CPU time "
            f"{'increased' if delta['cpu_seconds'] >= 0 else 'decreased'} from "
            f"{med['direct']['cpu_seconds']:.2f} s to "
            f"{med['certified']['cpu_seconds']:.2f} s "
            f"({delta['cpu_percent']:+.1f}%).")
    rss = []
    for entry in entries:
        change = entry["delta"]["peak_rss_bytes"]
        if change is None:
            continue
        gib = change / GIB
        rss.append(f"{'did not measurably change' if abs(gib) < 0.005 else f'changed by {gib:+.2f} GiB'} "
                   f"for {label(entry)} evaluation")
    if rss:
        parts.append("Process-lifetime peak host RSS " + ", and ".join(rss) +
                     "; under GPU execution the model weights reside in device memory "
                     "and are not counted in host RSS.")
    device = []
    for entry in entries:
        change = entry["delta"]["device_peak_bytes"]
        if change is None:
            continue
        device.append(f"{change / KIB:+,.0f} KiB for {label(entry)} evaluation")
    if device:
        parts.append("Median peak CUDA memory allocated changed by " +
                     ", and ".join(device) + ".")
    residuals = [e for e in entries if e["protocol_residual_seconds"] is not None]
    if residuals:
        parts.append(
            "Differencing the two arms cannot resolve the protocol's cost at this "
            "workload size: the run-to-run spread is larger than the quantity, and in "
            "these runs it is larger than it with the opposite sign. Measured instead "
            "inside the certified arm, as the wall time left after committing the "
            "artifacts and running the trials, the protocol accounts for " +
            ", and ".join(
                f"{e['protocol_residual_seconds']:.3f} s"
                f" ({100 * e['protocol_residual_seconds'] / e['median']['certified']['wall_seconds']:.3f}% "
                f"of the {label(e)} workload)" for e in residuals) +
            ", against a direct-arm residual of "
            + ", and ".join(f"{e['median']['direct']['other_seconds']:.3f} s"
                            for e in residuals) + ".")
    commits = [e for e in entries
               if e["median"]["direct"]["commitment_seconds"] is not None]
    if commits:
        parts.append(
            "Those deltas exclude the dominant cost, because both arms pay it: "
            "committing the provider's artifacts means hashing every provisioned file, "
            "which took " + ", and ".join(
                f"a median {e['median']['direct']['commitment_seconds']:.2f} s "
                f"against {e['median']['direct']['model_run_seconds']:.2f} s of actual "
                f"trials for the {label(e)} workload" for e in commits) +
            ". Commitment cost is linear in provisioned bytes and independent of the "
            "trial count, so it is amortized by larger test sets and is the term that "
            "dominates at small ones.")
    parts.append(
        "Certification transmitted " + ", and ".join(
            f"{entry['median_transmitted_bytes'] / KIB:.2f} KiB"
            f" for the {label(entry)} workload" for entry in entries) +
        " of certifier-workload protocol traffic on the in-process socket pair.")
    if all(entry["median_transmitted_control_bytes"] is not None for entry in entries):
        parts.append(
            "Almost all of that is one message: the sealed test environment, padded "
            "to a fixed length so its size reveals nothing about the secret test, "
            "accounts for " + ", and ".join(
                f"{entry['median_transmitted_environment_bytes'] / KIB:.2f} KiB"
                for entry in entries) + ", leaving " + ", and ".join(
                f"{entry['median_transmitted_control_bytes'] / KIB:.2f} KiB"
                f" of commit and certificate traffic for the {label(entry)} workload"
                for entry in entries) + ". The environment component is invariant to "
            "the workload and to the trial count by construction, so it is a property "
            "of the protocol rather than a per-evaluation cost.")
    return " ".join(parts)


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    entries = [workload(Path(argument)) for argument in sys.argv[1:]]

    runtime = entries[0]["runtime"]
    accelerator = runtime.get("accelerator") or {}
    instance = runtime.get("instance") or {}
    lines = [
        "# Cost of certification — Compute Engine L4",
        "",
        f"- instance: `{instance.get('machine_type')}` in `{instance.get('zone')}` "
        f"({instance.get('provisioning_model')}), "
        f"{accelerator.get('name')} {accelerator.get('total_memory_bytes', 0) / GIB:.0f} GiB, "
        f"driver {accelerator.get('driver')}, CUDA {accelerator.get('cuda_runtime')}",
        f"- torch {runtime['versions'].get('torch')}, "
        f"transformers {runtime['versions'].get('transformers')}, "
        f"python {runtime.get('python')}",
        "",
        "| workload | trials | repeats | wall direct | wall certified | Δwall | "
        "CPU direct | CPU certified | ΔCPU | GPU peak Δ | wire |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in entries:
        med, delta = entry["median"], entry["delta"]
        device = ("—" if delta["device_peak_bytes"] is None
                  else f"{delta['device_peak_bytes'] / KIB:+,.0f} KiB")
        lines.append(
            f"| {label(entry)} | {entry['n_trials']} | {entry['repeats']} | "
            f"{med['direct']['wall_seconds']:.2f} s | "
            f"{med['certified']['wall_seconds']:.2f} s | "
            f"{delta['wall_seconds']:+.2f} s ({delta['wall_percent']:+.1f}%) | "
            f"{med['direct']['cpu_seconds']:.2f} s | "
            f"{med['certified']['cpu_seconds']:.2f} s | "
            f"{delta['cpu_seconds']:+.2f} s ({delta['cpu_percent']:+.1f}%) | "
            f"{device} | {_wire_cell(entry)} |")

    if USD_PER_HOUR:
        rate = USD_PER_HOUR / 3600.0
        lines += ["", "## What it cost", "",
                  f"At ${USD_PER_HOUR:.4f}/hour for this instance "
                  f"(${rate:.6f} per second), list price.", "",
                  "| workload | commitment time | model run | protocol | "
                  "paired run | measurement set |",
                  "|---|---|---|---|---|---|"]
        campaign_total = 0.0
        for entry in entries:
            direct, certified = entry["median"]["direct"], entry["median"]["certified"]
            commit = direct["commitment_seconds"]
            if commit is None:
                continue
            # A process runs the evaluation twice, so the per-process figures double.
            per_process = (direct["wall_seconds"] + certified["wall_seconds"]) * rate
            campaign = entry["campaign_wall_seconds"] * rate
            campaign_total += campaign
            protocol = entry["protocol_residual_seconds"] or 0.0
            lines.append(
                f"| {label(entry)} | ${2 * commit * rate:.4f} | "
                f"${(direct['model_run_seconds'] + certified['model_run_seconds']) * rate:.4f} | "
                f"${protocol * rate:.6f} | ${per_process:.4f} | "
                f"${campaign:.2f} ({entry['processes']} paired runs) |")
        lines += ["", f"**Total for the measurement sets in this report: "
                      f"${campaign_total:.2f}.** Certification's own marginal cost is "
                      "the protocol column; the commitment column is paid by the "
                      "uncertified baseline too, since both arms commit.", ""]

    lines += ["", "## Where the time goes", "",
              "Both arms commit the provider's artifacts, so that cost cancels out of "
              "the direct-vs-certified delta above and is invisible in it — while "
              "remaining a real, and here dominant, cost of certifying at all.",
              "",
              "| workload | arm | commitment time | model run duration | protocol "
              "residual | commitment input | hash throughput |",
              "|---|---|---|---|---|---|---|"]
    for entry in entries:
        if entry["median"]["direct"]["commitment_seconds"] is None:
            lines.append(f"| {label(entry)} | — | not recorded | — | — | — | — |")
            continue
        artifacts = entry["commitment_input_bytes"]
        for phase in PHASES:
            med = entry["median"][phase]
            commit, evaluate = med["commitment_seconds"], med["model_run_seconds"]
            # Hashed twice: prepare_components records each component, then records it
            # again to detect a file changing while its loader consumed it.
            throughput = (f"{2 * artifacts / commit / 1e6:,.0f} MB/s (2x)"
                          if artifacts and commit else "—")
            spread = med.get("other_stdev_seconds")
            residual = (f"**{med['other_seconds']:.3f} s**"
                        + (f" ± {spread:.3f}" if spread is not None else ""))
            lines.append(
                f"| {label(entry) if phase == 'direct' else ''} | {phase} | "
                f"{commit:.2f} s | {evaluate:.2f} s | {residual} | "
                f"{(f'{artifacts / GIB:.2f} GiB' if artifacts else '—') if phase == 'direct' else ''} | "
                f"{throughput if phase == 'direct' else ''} |")

    lines += ["",
              "The direct arm's residual is the harness's zero point: it commits and "
              "evaluates and does nothing else. The certified arm's residual is the "
              "protocol itself — sealing, the socket transfer, signing and "
              "verification — measured inside one arm instead of by subtracting two "
              "multi-minute arms to find a sub-second quantity.",
              "", "## Reported paragraph", "", prose(entries), "",
              "## Per workload", ""]
    for entry in entries:
        parameters = entry["parameters"]
        lines += [
            f"### {label(entry)}",
            "",
            f"- environment `{entry['environment']}`, {entry['n_trials']} trials, "
            f"seed {entry['seed']}",
            f"- {entry['repeats']} repeats x 2 orderings = {entry['processes']} "
            f"processes, each running the evaluation twice",
            f"- model `{parameters.get('model') or parameters.get('program_model')}` "
            f"at {parameters.get('device')}/{parameters.get('dtype')}"
            + (f", attn `{parameters['attn_implementation']}`"
               if parameters.get("attn_implementation") else ""),
            f"- released m(FN)/m_fp = {entry['released']['m']:.4f} / "
            f"{entry['released']['m_fp']:.4f} — identical in both arms of every "
            f"process, which is the comparison's control, not a probe-quality result",
            f"- arms consistent across all processes: {entry['consistent']}",
        ]
        for problem in entry["disagreements"]:
            lines.append(f"  - **{problem}**")
        first = entry["first_arm_peak_rss_bytes"]
        lines += [
            f"- first-arm host peak RSS: direct {first['direct'] / GIB:.3f} GiB, "
            f"certified {first['certified'] / GIB:.3f} GiB",
            "",
        ]
    lines += [
        "## Glossary",
        "",
        "Terms as used above, stated precisely because the measurement design depends",
        "on the distinctions.",
        "",
        "**Quantities**",
        "",
        "| term | field | definition |",
        "|---|---|---|",
        "| commitment input | `commitment_input_bytes` | Total bytes fed to the "
        "certificate commitment — every provisioned artifact file that gets hashed. "
        "Fixed by the model and probe, independent of the test set. |",
        "| commitment time | `commitment_seconds` | Wall time to compute the "
        "certificate commitment over those bytes. Hashed twice per arm (once, then "
        "again to detect a file changing while its loader consumed it). |",
        "| model run duration | `model_run_seconds` | Wall time executing the trials: "
        "the agent generating and the tap scoring. Scales with the trial count. |",
        "| protocol residual | wall − commitment − model run | The protocol's own "
        "work: sealing, socket transfer, signing, verification. Measured inside one "
        "arm rather than by differencing arms. |",
        "| transmitted total | `transmitted_bytes_total` | All bytes crossing the "
        "certifier/workload socket pair. **Not** a compressed certificate — see the "
        "split below. |",
        "| ↳ environment | `transmitted_environment_bytes` | The secret test "
        "environment sent **to** the workload, sealed and padded to a fixed length so "
        "its size leaks nothing. ~99.7% of the total, and invariant to workload and "
        "trial count. |",
        "| ↳ commit | `transmitted_commit_bytes` | The commit message from workload to "
        "certifier, carrying the commitment digest `h_c` (32 bytes) plus signature. |",
        "| ↳ certificate | `transmitted_certificate_bytes` | The certificate itself, "
        "workload to certifier: `(m, m_fp)`, `h_c`, `h_s`, `t_E`, signature. |",
        "",
        "**Measurement structure**",
        "",
        "| term | definition |",
        "|---|---|",
        "| arm | One execution of the evaluation, either `direct` (no protocol) or "
        "`certified` (the same evaluation driven through the "
        "trusted-workload/certifier protocol). |",
        "| **paired run** | One OS process that loads the model **once**, then runs "
        "both arms back to back. The shared model load is what makes the two arms "
        "comparable; each paired run yields one paired observation. Previously called "
        "\"one process\". |",
        "| arm order | Which arm runs first inside a paired run — `direct_first` or "
        "`certified_first`. Counterbalanced so once-per-process warm-up (CUDA context, "
        "allocator, autotuning) does not land on one arm. |",
        "| repeat | Index of a paired run within one arm order. |",
        "| **measurement set** | All paired runs sharing one workload configuration "
        "(model, dtype, trial count, seed, token budget): `2 × repeats` paired runs. "
        "Medians are taken over this set. Previously called \"campaign\". |",
        "| session | Everything run between one instance start and stop. Relevant "
        "because a stop/start can reschedule onto different physical hardware, which "
        "would confound pooled timing data. |",
        "",
        "## Scope",
        "",
        "- Not a TEE measurement. The certifier and the trusted workload are two "
        "protocol endpoints in one process on a general-purpose VM; attestation and "
        "confidential provisioning are not part of this.",
        "- `wire` is the exact byte count on that in-process socket pair, not a "
        "network measurement. It is expected to be near-constant across workloads: "
        "the environment crosses the boundary padded to a fixed length so its size "
        "leaks nothing, and that padding dominates the total.",
        "- Host RSS excludes device memory; only the first arm of each process has a "
        "usable high-water mark.",
        "- `(m, m_fp)` appears as the control that both arms agree, not as a result. "
        "For the financial workload the threshold is uncalibrated and the concealment "
        "label is the repo's heuristic judge.",
    ]

    report = "\n".join(lines) + "\n"
    destination = Path(entries[0]["directory"])
    (destination / "report.md").write_text(report)
    (destination / "report.json").write_text(
        json.dumps({"schema_version": "ct-cost-report/1", "workloads": entries},
                   indent=2, sort_keys=True) + "\n")
    print(report)
    print(f"wrote {destination / 'report.md'} and {destination / 'report.json'}")
    return 0 if all(entry["consistent"] for entry in entries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
