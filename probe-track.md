# Probe track (Apollo linear probes)

The probe tap uses **Apollo Research's `deception-detection`** at the pinned
`third_party/deception-detection` submodule commit, plus our Gemma-4 patch. The RUNBOOK
documents the training walkthrough; `ProbeMonitor` loads the resulting artifact.

- Repo: https://github.com/ApolloResearch/deception-detection
- Paper: Goldowsky-Dill et al., "Detecting Strategic Deception with Linear Probes,"
  ICML 2025 (arXiv:2502.03407)
- **License: none.** The repo ships no LICENSE. Contact Apollo before redistributing
  or packaging its source or derived artifacts.

This file is the reference (provenance, license, and how the trained probe is used in
code). The operational walkthrough to train one lives in the RUNBOOK, so there is a
single set of steps to maintain.

## Gemma-4 support (`third_party/patches/gemma4-port.patch`)

Apollo's published pipeline supports Gemma-2 / Llama / Mistral only. The patch adds the
Gemma-4 family: the `ModelName` entry and loader, the Gemma-4 chat-template
detection-mask regex (`<|turn>`/`<turn|>`, native system turn), the assistant-header
padding, and a `cut_at_layer` fix for Gemma-4's multimodal layer nesting. It also
switches the forward kwarg renamed in transformers 5, which is why Gemma-4 support and
Apollo's original `transformers<5` pin cannot coexist.

It ships as a patch rather than a fork or a vendored copy **because Apollo's repository
has no LICENSE**: we redistribute only our own diff. `git apply` it per the RUNBOOK;
`evals/financial/test_gemma4_patch.py` fails if the submodule pin moves out from under
it, and asserts the patch touches only the four Apollo files its header documents.

Verified without weights (Python 3.14 / torch 2.13 / transformers 5.15): the `ModelName`
plumbing resolves to 60 text layers, the tokenizer loads via `GEMMA4_TOKENIZER_PATH`,
and Apollo's own `TokenizedDataset` masking marks exactly the assistant turn
(`<|turn>model\nThe answer is 4.<turn|>`) with the prompt excluded. A full training run
on real Gemma-4 weights has **not** been executed here.

## Reading the thinking channel (`evals/financial/thinking_rollouts.py`)

Our own module (no Apollo code, hence in-repo and unit-tested rather than in the patch).
Apollo's datasets score detect-tokens inside a rendered dialogue; a Gemma-4 rollout with
thinking on instead emits a thought channel and then a response, and the probe should
read exactly those generated tokens. `build_thinking_tokenized_dataset` takes
pre-tokenized `(prompt_ids, generation_ids)` records, left-pads, and builds a detection
mask true over exactly the generation. The result is drop-in for `Activations.from_model`
and the `DirectionDetector` scoring path, so no Apollo file changes to consume it.

This is a **different measurement** from the one `ProbeMonitor` performs at
certification time (below), which reads the answer positions and not the hidden
reasoning. The two must be reconciled before their numbers are comparable.

## Train a probe

The full step-by-step — the single-venv install, the Gemma-4 patch, the Hugging Face
token, and the `experiment.py` training command —
is in **[RUNBOOK Phase 2](RUNBOOK.md#phase-2--probe-track)**. It writes
`results/repe_lr__..._<model>__<timestamp>/detector.pt`, a pickle dict
`{layers, directions, scaler_mean, scaler_scale, normalize, reg_coeff}`. The RepE
config (`repe.yaml`) trains on `repe_honesty__plain`, a fully-local paired
honest/deceptive set (a CSV, no rollouts or API); other shipped train sets are
`roleplaying`, `insider_trading` (variant `upscale`), and `werewolf`.

## Open gap: the eval program is still Gemma-2

The patch makes it possible to *train* on Gemma 4; it does not switch the thing being
probed. `evals/financial/program.py` still runs `Gemma2Program`, and the defaults in
`evals_runner/taps.py` and `evals_runner/protocol_integrity_exp.py` still name
`google/gemma-2-2b-it`. Since a probe is only valid on the exact weights whose
activations trained it, **a Gemma-4-trained probe cannot be certified through the
financial slice until that program is replaced.** Their docstrings gave "Apollo does not
support Gemma-4" as the reason for the Gemma-2 choice; that reason is now gone, but the
replacement is real work (Gemma 4 is a multimodal `Gemma4ForConditionalGeneration` whose
text layers nest differently, and the layer indices the probe reads must be re-derived),
so it is called out here rather than half-done.

## Use the probe as a tap

Accurate to the code as it stands today — i.e. Gemma-2, per the gap above.

```python
from ct_core.agent import Agent
from evals.financial.actions import FINANCIAL_TOOLS
from evals.financial.agent import InsiderTradingControl
from evals.financial.probe_tap import ApolloProbe, ProbeMonitor
from evals.financial.program import Gemma2Program

PROBED_MODEL_ID = "<PROBED_MODEL_ID>"
DETECTOR_PATH = "<PATH_TO_MATCHING_DETECTOR_PT>"
THRESHOLD = 0.0  # replace with a cutoff calibrated on a disjoint benign split

probe = ApolloProbe.load(DETECTOR_PATH)
agent = Agent(Gemma2Program(PROBED_MODEL_ID),   # exact weights used to train this probe
              InsiderTradingControl(), FINANCIAL_TOOLS)
tap = ProbeMonitor(probe, threshold=THRESHOLD)
# run the workload with this (agent, tap): the program returns activations at
# `tap.probe_layers`, which the workload passes to the probe.
```

The runnable financial command and its options live in
[RUNBOOK Phase 2](RUNBOOK.md#phase-2--probe-track).

`threshold` is the control-calibrated cutoff. Apollo calibrates on `alpaca` at a
target FPR (`threshold = nanquantile(control_scores, 1 - fpr)`, default fpr 0.01);
reproduce that on a benign split, or start at 0.0 and calibrate.

The probe reads activations returned by the same committed `Gemma2Program.generate`
invocation that produced the response: it retains each selected layer's decision state
from the generation step that emitted each token, without replaying the model. The
scoring mirrors Apollo's `LogisticRegressionDetector.score` followed by a mean over
tokens. Apollo instead scores its datasets' detect-token states within the full rendered
dialogue; this alignment difference must be reconciled before trusting the probe's
numbers.

## Prior exploratory results (Gemma-4-31B, thinking pipeline)

Recorded here because the work happened outside version control and its artifacts are
already gone: the rollouts, the hand-added natural-deception labels, and the trained
`probe_lr.pt` / `probe_mms.pt` no longer exist, and the per-experiment AUROCs were
printed to stdout that was not kept. **Treat all of this as direction, not evidence** —
none of it is reproducible from this repo today.

- **Layer choice.** A sweep trained a per-layer LR probe on instructed roleplay and
  scored a natural (un-instructed) test set, over layers 1–19 of 60. Every script
  written after the sweep hardcodes **layer 8** (~13% depth), so that is what it
  selected. Relevant when overriding `repe.yaml`'s Gemma-2 `detect_layers: [22]`, but a
  different training set — re-run the sweep rather than copying the number.
- **Instructed → natural transfer.** A probe trained only on *instructed* deception
  (labels by construction, no grader) scored un-instructed rollouts where the model
  chose to deceive on its own. Baselines: instructed-honest ≈ **−6.1**,
  instructed-deceptive ≈ **+2.9**; eight natural deceptive rollouts scored **+0.68 to
  +6.50**, several above the instructed-deceptive mean. The probe saw only model-emitted
  text, never the scenario.
- **Domain taxonomy.** Five domains were generated (roleplay, insider trading,
  sandbagging, werewolf, RepE bare factual lying) to test whether training breadth fixes
  cross-domain *calibration* rather than just AUROC, including leave-one-domain-out. The
  designs survive in `../certified-probe/train_*.py`; the results do not.

## Status

`ProbeMonitor`'s scoring math and pickle loader are unit-tested (against a real example
probe and a synthetic one). A one-trajectory certification smoke test has also run on
real matching weights with a trained probe; reproduce or extend it by filling the
Runbook's checkpoint and detector placeholders and training your own probe per
[RUNBOOK Phase 2](RUNBOOK.md#phase-2--probe-track).

The Gemma-4 patch and `thinking_rollouts.py` are unit-tested and validated against
Apollo's own masking code without weights. Neither has been through a real Gemma-4
training run here, and the certification path still runs Gemma-2 (see the open gap
above).
