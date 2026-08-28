# `util/` — shared building blocks used by more than one eval

The pieces the evals share: `dynamic_test_environment` (the static test environment
`E_D`), `treecut` (the answerable/unanswerable item generator), and `model_backend` (the
Gemma / fake LLM backend). Purpose-specific things (the TreeCut environments, the Apollo
probe) live under [`evals/`](../evals), not here.

None of these produce output on their own; they come alive when an eval runs. Two
behaviors matter downstream: the environment's labels are fixed independently of the
program's answers, so the score is the environment's ground truth and not the provider's,
and the backend's `digest()` records the checkpoint revision when it can resolve one.

## References

- TreeCut (arXiv:2502.13442, [j-bagel/treecut-math](https://github.com/j-bagel/treecut-math), Apache-2.0).
- Gemma 4 (Google DeepMind, Apache-2.0), using the compatible Hugging Face checkpoint
  selected through the Runbook's `<PROGRAM_MODEL_ID>` / `<MONITOR_MODEL_ID>` placeholders.

## Run it

Nothing to run directly. These come alive when a real model runs; see
[RUNBOOK Phase 1](../RUNBOOK.md#phase-1--real-model-slice-the-cot-confidence-tap-local).
