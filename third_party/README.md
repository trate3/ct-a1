# `third_party/` — pinned upstream code, imported not copied

External dependencies live here as git submodules pinned to specific upstream commits.
Populate them per RUNBOOK Phase 0; Phase 2 documents the local Apollo loader changes.

## How it works

- A submodule SHA identifies the upstream base version without copying it into our tree.
- Siblings load that code in place instead of duplicating it. `util/treecut` puts
  the `treecut-math` package on `sys.path` and calls its `gen_data.generate_qa`
  unchanged to build the unanswerable-math environment.
- `evals/financial` uses `deception-detection` for the white-box probe track,
  wiring an Apollo linear probe into a `ProbeMonitor`. See `probe-track.md` at the
  repo root for training the probe and fetching its data.
- A pinned SHA identifies upstream source; it does not by itself establish experimental
  correctness. Record local submodule changes, model revisions, probe hashes,
  thresholds, and evaluation configuration with final results.

## References

- [j-bagel/treecut-math](https://github.com/j-bagel/treecut-math) — Apache-2.0,
  arXiv:2502.13442. The unanswerable-math question generator.
- [ApolloResearch/deception-detection](https://github.com/ApolloResearch/deception-detection)
  — no repository license, arXiv:2502.03407. The linear-probe repo behind the probe
  track; see `probe-track.md` and RUNBOOK Phase 2.

## Run it

This folder is a library of pinned code with no run command of its own. It is
exercised through `util/treecut` (math environment) and `evals/financial` (probe
track). To fetch the submodules, follow
[RUNBOOK Phase 0 (submodule init) and Phase 2 (Apollo)](../RUNBOOK.md#phase-0--get-the-mock-harness-running-5-min-no-models).
