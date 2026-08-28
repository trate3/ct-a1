# `evals/confidence/` — CoT-confidence tap vs black-box

This case tests whether a monitor reading the program's chain of thought catches
failures missed by an answer-only monitor, and whether certification preserves the
measured losses.

## What you'll see

`python -m evals_runner taps confidence` prints a signed certificate with the tap's
aggregate losses `(m, m_fp)`, then a separation line comparing the tap against the
black-box comparator on the same items. Two things to read:

- **Separation.** A positive `comparator_FN - tap_FN` means the CoT tap catches misses
  the answer-only monitor lets through; zero or negative separation is also a valid
  empirical result. The gap
  lives or dies on the test set. On items whose difficulty shows on the prompt surface
  there is nothing in the thought channel to read, and small models reveal little in
  simple QA. TreeCut is used precisely because unanswerability requires working the
  problem. If the gap is flat, vary the model and item difficulty, then choose the
  sample size from the target uncertainty before concluding anything.
- **Preservation.** On frozen inputs the certified `(m, m_fp)` should equal what the
  direct run produces. The comparison command rejects a result if they differ; fresh
  model nondeterminism can also cause a mismatch.

`AdaptiveEnvironment` implements transcript-conditioned test generation, but the
current confidence CLI uses `TreeCutEnvironment` rather than selecting that path.

## References

- TreeCut (arXiv:2502.13442), the answerable/unanswerable generator behind the test
  environments; see [`util/treecut`](../../util/treecut).
- The compatible program and monitor checkpoints selected for the study; the Runbook
  marks both model IDs as explicit placeholders rather than fixing their size.

## Run it

The command and its options are in
[RUNBOOK Phase 1](../../RUNBOOK.md#1c-run-the-slice) (1c to run, 1d for what to expect).
Real numbers need the `[model]` extra and downloaded weights.
