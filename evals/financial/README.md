# `evals/financial/` — deception-probe tap, white-box (Apollo)

The white-box counterpart to the confidence slice: the same certified message flow,
but the tap is a trained Apollo linear probe reading the program's activations
instead of text, and the secret test is Apollo's interactive insider-trading scenario.

## What you'll see

`python -m evals_runner taps financial` runs an in-process certified session over Apollo's
insider-trading scenario as a secret, action-dependent environment. The committed Gemma-2 agent owns its conversation state
and chooses among the versioned tool interfaces declared by Apollo's scenario variants.
The environment executes the selected test-side tool simulation and returns its result;
it never forces a trade or report. The
probe reads the activations from those actual decisions. The run prints a signed
certificate with the probe's `(m, m_fp)`: missed-positive and false-positive trial
counts, respectively, each divided by all fresh trials. It then re-verifies the
signature and public specification binding.

The certificate is the framework property: the committed agent and probe (weights +
threshold, bound in `h_c`) certified against the certifier's secret insider-trading environment.
Probe accuracy, threshold calibration, and label quality are separate empirical
questions from certificate validity. The environment currently labels concealment with
a heuristic judge. It runs on CPU (Gemma-2 can't do bf16 on MPS), so it is slow —
keep `--n-trajectories` small.

The workload image registers the insider-trading driver and its action contract. C's
encrypted package contains only inert scenario data plus the exact registered source;
each scenario runs as a fresh two-step environment trial.

The numpy scoring math and the pickle loader (against a shipped example probe) are
unit-tested; the end-to-end run needs the gated weights and the submodule, so it is not
in CI.

`python -m evals_runner secret-test-gaming-exp --case financial` runs the bounded
public/reused-hidden/fresh-secret adaptation study over this same agent and probe; see
RUNBOOK Phase 2, Step C for its required model and detector arguments.
`python -m evals_runner model-agent-divergence-exp` runs the committed control-logic
divergence sweep and non-certified structural-audit arm described in Step D.

## References

- Apollo's "Detecting Strategic Deception with Linear Probes" (arXiv:2502.03407),
  trained from the pinned `third_party/deception-detection` source plus the Gemma-4
  patch in `third_party/patches/`. The repository ships no OSS license.
- The insider-trading scenario (Scheurer et al. arXiv:2311.07590), the secret test
  environment (its scenario prompts ship in the submodule).
- Training the probe is [RUNBOOK Phase 2](../../RUNBOOK.md#phase-2--probe-track).
  [`probe-track.md`](../../probe-track.md) covers Apollo's provenance/license, the
  `ProbeMonitor` usage snippet, and threshold calibration. The Gemma-2 adapter is
  `evals/financial/program.py`; `thinking_rollouts.py` builds the training-time detection
  mask over a Gemma-4 thought channel.

## Run it

The full step-by-step (train the probe, then certify it end to end) is
[RUNBOOK Phase 2](../../RUNBOOK.md#phase-2--probe-track).
