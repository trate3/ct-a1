#!/usr/bin/env python3
"""Reconstruct what this instance has cost, from its own start/stop history.

Usage: spend.py [--instance ct-probe-l4] [--zone us-west1-a]

Compute Engine records insert/start/stop/delete as operations on the instance, so
billable uptime can be rebuilt after the fact without having set up a billing export
first. RUNNING hours are charged at the full instance rate; a STOPPED instance still
bills for its disk, which is the part that is easy to forget and the reason a
"finished" experiment can keep costing money.

List prices, so this is an upper bound: sustained-use discounts, credits and free tier
all reduce the real invoice, and none of them are visible here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import pricing  # noqa: E402


def gcloud(args: list[str]) -> str:
    return subprocess.run(["gcloud", *args], capture_output=True, text=True,
                          check=True).stdout


def parse_time(value: str) -> dt.datetime:
    """Parse to UTC. Compute Engine reports operation times in the caller's local
    offset (-07:00 here), so formatting the parsed value directly prints a mix of
    local and UTC clocks in one table."""
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(dt.timezone.utc)


def history(project: str, instance: str, zone: str) -> list[tuple[dt.datetime, str]]:
    raw = gcloud(["compute", "operations", "list", "--project", project,
                  "--filter", f"targetLink~{instance}$ AND zone~{zone}",
                  "--format", "json"])
    events = []
    for op in json.loads(raw):
        kind = op.get("operationType", "")
        when = op.get("endTime") or op.get("insertTime")
        if not when:
            continue
        if kind in ("insert", "start"):
            events.append((parse_time(when), "running"))
        elif kind in ("stop", "delete"):
            events.append((parse_time(when), "stopped" if kind == "stop" else "gone"))
    return sorted(events)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="certified-taps")
    parser.add_argument("--instance", default="ct-probe-l4")
    parser.add_argument("--zone", default="us-west1-a")
    parser.add_argument("--disk-gib", type=int, default=200)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rates = pricing.rates(args.zone.rsplit("-", 1)[0], 8, 32, 1, args.disk_gib)
    events = history(args.project, args.instance, args.zone)
    if not events:
        print(f"no operations found for {args.instance} in {args.zone}", file=sys.stderr)
        return 1

    now = dt.datetime.now(dt.timezone.utc)
    spans, state, since = [], None, None
    for when, new_state in events + [(now, "now")]:
        if state in ("running", "stopped") and since is not None:
            spans.append((state, since, when, (when - since).total_seconds() / 3600))
        if new_state == "gone":
            state, since = None, None
        elif new_state != "now":
            state, since = new_state, when

    running = sum(h for s, _, _, h in spans if s == "running")
    stopped = sum(h for s, _, _, h in spans if s == "stopped")
    compute_cost = running * rates["compute_usd_per_hour"]
    disk_cost = (running + stopped) * rates["hourly"]["disk"]
    ip_cost = running * rates["hourly"]["ip"]
    total = compute_cost + disk_cost + ip_cost

    if args.json:
        print(json.dumps({"rates": rates, "running_hours": running,
                          "stopped_hours": stopped, "compute_usd": compute_cost,
                          "disk_usd": disk_cost, "ip_usd": ip_cost,
                          "total_usd": total}, indent=2, sort_keys=True))
        return 0

    print(f"{args.instance} in {args.zone}\n")
    for state, start, end, hours in spans:
        print(f"  {start:%Y-%m-%d %H:%M} -> {end:%H:%M}Z {state:<8} {hours:6.2f} h")
    print(f"\n  running {running:6.2f} h  x ${rates['compute_usd_per_hour']:.4f}/h "
          f"(8 vCPU + 32 GiB + L4)  = ${compute_cost:7.2f}")
    print(f"  disk    {running + stopped:6.2f} h  x ${rates['hourly']['disk']:.4f}/h "
          f"({args.disk_gib} GiB balanced)      = ${disk_cost:7.2f}")
    print(f"  ip      {running:6.2f} h  x ${rates['hourly']['ip']:.4f}/h"
          f"                        = ${ip_cost:7.2f}")
    print(f"  {'TOTAL':<48}= ${total:7.2f}")
    print(f"\n  {rates['caveat']}")
    print(f"  stopped but not deleted still costs "
          f"${rates['stopped_usd_per_hour'] * 24:.2f}/day")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
