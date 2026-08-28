"""Price measured direct-versus-certified runs in dollars (RQ1).

Reads any number of `run_comparison` result JSONs (whatever
`certification-overhead-exp` or `taps ... --certification-overhead-out` wrote)
and reports the two charges separately: protocol overhead and TEE premium.
See evals_runner/util/cost_model.py for what those mean and what they exclude.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .util.cost_model import cost_breakdown, format_table, load_pricing


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "results", nargs="+",
        help="result JSON files, or directories to scan for *.json",
    )
    parser.add_argument(
        "--instance", default="a3-highgpu-1g",
        help="priced machine shape from the pricing table (default: a3-highgpu-1g)",
    )
    parser.add_argument(
        "--pricing", default=None,
        help="pricing table JSON (default: evals_runner/util/pricing_gcp.json)",
    )
    parser.add_argument("--out", default=None, help="write the breakdown JSON here")
    parser.add_argument(
        "--on-demand", action="store_true",
        help="price against on-demand instead of Spot. Spot is the default because "
             "confidential H100 on A3 is Spot/flex-start only, so Spot is the honest "
             "baseline to compare a confidential run against.",
    )
    parser.set_defaults(run=run)


def _collect(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(path.rglob("*.json")))
        else:
            found.append(path)
    return found


def run(args: argparse.Namespace) -> int:
    pricing = load_pricing(args.pricing)
    breakdowns = []
    skipped = []
    for path in _collect(args.results):
        try:
            result = json.loads(path.read_text())
        except json.JSONDecodeError:
            skipped.append((path, "not valid JSON"))
            continue
        if result.get("schema_version") != "ct-eval/1":
            skipped.append((path, "not a direct-vs-certified result"))
            continue
        breakdown = cost_breakdown(
            result, pricing, args.instance, use_spot=not args.on_demand
        )
        breakdown["source"] = str(path)
        breakdowns.append(breakdown)

    if not breakdowns:
        print("no direct-vs-certified results found")
        for path, why in skipped:
            print(f"  skipped {path}: {why}")
        return 1

    print(format_table(breakdowns))
    # Silence is not evidence of coverage: say what was passed over.
    for path, why in skipped:
        print(f"skipped {path}: {why}")

    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {"schema_version": "ct-cost/1", "instance": args.instance,
                 "breakdowns": breakdowns},
                indent=2, sort_keys=True,
            )
            + "\n"
        )
        print(f"\nwrote {destination}")
    return 0
