"""Run an evaluation or local service with ``python -m evals_runner``."""

from __future__ import annotations

import argparse

from . import (
    certification_overhead_exp,
    cost_report_exp,
    model_agent_divergence_exp,
    protocol_integrity_exp,
    secret_test_gaming_exp,
    taps,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m evals_runner")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, module, help_text in (
        (
            "certification-overhead-exp",
            certification_overhead_exp,
            "measure direct-versus-certified overhead",
        ),
        ("cost-report", cost_report_exp, "price measured runs in dollars (RQ1)"),
        ("taps", taps, "run a concrete tap evaluation"),
        (
            "model-agent-divergence-exp",
            model_agent_divergence_exp,
            "compare model-local evidence with whole-agent behavior and code audit",
        ),
        (
            "secret-test-gaming-exp",
            secret_test_gaming_exp,
            "run the bounded secret-test gaming experiment",
        ),
        (
            "protocol-integrity-exp",
            protocol_integrity_exp,
            "compare honest execution with environment-message substitution",
        ),
    ):
        command = commands.add_parser(name, help=help_text)
        module.configure_parser(command)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
