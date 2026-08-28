# `util/model_backend/` — the LLM backend

The seam between the harness and a concrete LLM. `TransformersBackend` runs Gemma 4;
`FakeBackend` scripts the same calls so everything else is testable with no GPU. Swapping
between them changes no protocol code.

`digest()` records the model identifier and a supplied or resolved checkpoint revision.
If neither is available, it records `unpinned`; the current CLI has no `--revision`
option. Real execution needs the `[model]` extra.

## References

- Gemma 4 (Google DeepMind, Apache-2.0), using the compatible Hugging Face checkpoint
  selected through the Runbook's `<PROGRAM_MODEL_ID>` / `<MONITOR_MODEL_ID>` placeholders.

## Run it

Nothing to run directly. Used by [`evals/confidence`](../../evals/confidence); model setup is in
[RUNBOOK Phase 1](../../RUNBOOK.md#phase-1--real-model-slice-the-cot-confidence-tap-local).
