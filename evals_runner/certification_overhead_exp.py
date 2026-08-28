"""Run and record the GPU-free certification-overhead experiment."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from evals_runner.util import ExperimentConfig, run_comparison, write_result
from evals.basic_mocked import FIXTURE
from evals.basic_mocked.mocked_protocol_components import (
    MockAgentDriver,
    MockTapDriver,
    agent_package,
    build_environment,
    tap_package,
)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", default=str(FIXTURE))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--certified-first", action="store_true", help="run the certified arm first"
    )
    parser.add_argument(
        "--out", default="evals_results/certification_overhead_exp/basic_mocked.json"
    )
    parser.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
    data = Path(args.data)
    config = ExperimentConfig(
        experiment="basic-mocked-direct-vs-certified",
        seed=args.seed,
        n_trials=args.n,
        environment="static-treecut",
        parameters={"data_sha256": hashlib.sha256(data.read_bytes()).hexdigest()},
    )
    result = run_comparison(
        config,
        build_agent_package=lambda: agent_package(data),
        build_tap_package=tap_package,
        agent_drivers=(MockAgentDriver,),
        tap_drivers=(MockTapDriver,),
        build_environment=lambda: build_environment(data, args.n, args.seed),
        certified_first=args.certified_first,
    )
    destination = write_result(result, args.out)
    print(f"wrote {destination} "
          f"(transmitted: {result['overhead']['transmitted_bytes_total']:,} bytes)")
    return 0
