# `util/dynamic_test_environment/` — the certifier's secret test environment

`StaticEnvironment`, the non-adaptive test environment `E_D`, holds a fixed labeled item
set at the certifier. C provisions the environment once; the trusted workload executes
its queries and labels internally. Those labels are fixed independently of the
program's answers, so the score is the environment's ground truth rather than anything
the provider controls.

On the protocol path, C sends a canonical package containing these items and the exact
`StaticEnvironment` source. The trusted workload authenticates the package, matches the
source to its registered driver, and creates a fresh one-step environment instance for
each trial.

Dynamic environments (fresh coins per run, or queries conditioned on the transcript)
share the same interface but are purpose-specific, so they live with their eval. See the
TreeCut `E_gen` / `E_adapt` in
[`evals/confidence/environments.py`](../../evals/confidence/environments.py) and the
action-conditioned insider-trading environment in
[`evals/financial/insider_trading_env.py`](../../evals/financial/insider_trading_env.py).

## Run it

Nothing to run directly. It is exercised in
[RUNBOOK Phase 0](../../RUNBOOK.md#phase-0--get-the-mock-harness-running-5-min-no-models).
