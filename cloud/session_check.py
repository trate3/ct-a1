#!/usr/bin/env python3
"""Compare timing batches taken in different sessions before trusting them together.

Usage: session_check.py <run-dir> <run-dir> [...]

Stopping and restarting a Compute Engine instance can schedule it onto different
physical hardware, so two batches of wall-clock measurements taken either side of a
stop are not automatically comparable -- pooling them, or reading a trend across them,
can turn a hardware change into a fake result.

Two independent checks:

  1. Hardware identity. The GPU UUID and CPU model are recorded per run. Same UUID
     means the same physical card, which settles the question directly.

  2. commit_artifacts as a calibration probe. Hashing the provisioned files is
     independent of the trial count and depends only on the artifacts and the host, so
     two runs of the same workload at different `n` should agree on it. A discrepancy
     there is a host effect, not a workload effect -- which makes it a usable control
     even when the runs are otherwise not poolable.
"""

from __future__ import annotations

import glob
import json
import re
import statistics
import sys
from pathlib import Path


def load(directory: str) -> dict:
    runs = []
    for f in sorted(glob.glob(directory + "/*_r*.json")):
        if re.search(r"_r\d+\.json$", f):
            runs.append(json.loads(Path(f).read_text()))
    if not runs:
        raise SystemExit(f"no result files under {directory}")
    reference = runs[0]
    accelerator = reference["runtime"].get("accelerator") or {}
    commits = [r[arm]["metrics"]["phases"]["commit_artifacts_seconds"]
               for r in runs for arm in ("direct", "certified")
               if (r[arm]["metrics"].get("phases") or {}).get("commit_artifacts_seconds")]
    return {
        "dir": directory,
        "case": reference["config"]["experiment"].replace("-certification-overhead", ""),
        "n_trials": reference["config"]["n_trials"],
        "artifact_bytes": reference["config"]["parameters"].get("artifact_bytes"),
        "gpu_uuid": accelerator.get("uuid"),
        "gpu_name": accelerator.get("name"),
        "driver": accelerator.get("driver"),
        "cpu": reference["runtime"].get("cpu_model"),
        "instance": (reference["runtime"].get("instance") or {}).get("instance_id"),
        "commit_median": statistics.median(commits) if commits else None,
        "commit_stdev": statistics.stdev(commits) if len(commits) > 1 else None,
        "processes": len(runs),
    }


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    batches = [load(a) for a in sys.argv[1:]]

    print(f"{'run':<52}{'case':<12}{'n':>4}{'proc':>6}{'commit s':>10}{'±':>8}")
    for b in batches:
        commit = "—" if b["commit_median"] is None else f"{b['commit_median']:.2f}"
        spread = "—" if b["commit_stdev"] is None else f"{b['commit_stdev']:.2f}"
        print(f"{Path(b['dir']).name:<52}{b['case']:<12}{b['n_trials']:>4}"
              f"{b['processes']:>6}{commit:>10}{spread:>8}")

    print("\nhardware")
    for b in batches:
        print(f"  {Path(b['dir']).name}")
        print(f"    gpu  {b['gpu_name']}  uuid={b['gpu_uuid']}  driver={b['driver']}")
        print(f"    cpu  {b['cpu']}")
    uuids = {b["gpu_uuid"] for b in batches if b["gpu_uuid"]}
    unknown = [b for b in batches if not b["gpu_uuid"]]
    if not uuids:
        print("  VERDICT: no GPU UUID recorded in any batch (they predate this")
        print("           instrumentation); the commitment probe below is the only "
              "evidence.")
    elif len(uuids) > 1:
        print("  VERDICT: DIFFERENT physical GPUs — do not pool wall-clock data "
              "without saying so.")
    elif unknown:
        # One UUID plus batches that recorded none is not evidence they agree.
        print(f"  VERDICT: {len(batches) - len(unknown)}/{len(batches)} batches "
              f"recorded a UUID, all {uuids.pop()}.")
        print(f"           {len(unknown)} batch(es) recorded none, so hardware "
              "identity is UNCONFIRMED for those;")
        print("           the commitment probe below is the evidence that covers them.")
    else:
        print(f"  VERDICT: same physical GPU across all batches ({uuids.pop()}).")

    print("\ncommit_artifacts probe (same artifacts should cost the same, whatever n)")
    groups: dict[tuple, list] = {}
    for b in batches:
        if b["commit_median"] is not None:
            groups.setdefault((b["case"], b["artifact_bytes"]), []).append(b)
    compared = False
    for (case, _), members in groups.items():
        if len(members) < 2:
            continue
        compared = True
        values = [m["commit_median"] for m in members]
        low, high = min(values), max(values)
        spread = 100 * (high - low) / low
        print(f"  {case}: " + ", ".join(f"n={m['n_trials']} -> {m['commit_median']:.2f}s"
                                        for m in members))
        print(f"    spread {spread:.1f}%  "
              + ("consistent; the host looks unchanged" if spread < 5 else
                 "MATERIAL — treat the batches as separate sessions"))
    if not compared:
        print("  (only one batch per workload; nothing to compare directly)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
