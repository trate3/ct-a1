# Runbook: from mock harness to fully running (local first, GCP later)

Step-by-step to take this repo from the GPU-free mock harness to real model runs
on your own machine, then later to Google Confidential Space. **No cloud is needed
for anything through Phase 2.**

> **Honesty note.** Phase 0 and the local three-process mock rehearsal are verified in
> this repo. Real-weight commands depend on the selected checkpoints and available
> memory; they validate wiring when run, not the paper's eventual effect sizes. The
> full-scale Phase 1 and Phase 2 studies have not been executed here.

---

## Phase 0 — get the mock harness running (5 min, no models)

Confirms the protocol works end to end before any model downloads.

```bash
git clone --recursive https://github.com/mychern/certified_taps
cd <repo root>                   # the folder that contains pyproject.toml
# cloned without --recursive?  git submodule update --init

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest -q                        # expect: all tests pass; optional probe-loader checks may skip
python -m evals_runner protocol-integrity-exp e2e             # CERTIFIED + signature VALID + separation
python -m evals_runner protocol-integrity-exp e2e --tamper-environment  # certifier rejects substituted c_E
python -m evals_runner secret-test-gaming-exp             # gamed tap aces public, fails the secret test
python -m evals_runner certification-overhead-exp         # writes evals_results/certification_overhead_exp/basic_mocked.json
```

The `e2e` command performs local attestation setup before launching the workload:
it pins `(pk_TEE, m_ref)` in `.keys/attestation.json`. For a manual multi-process
rehearsal, run `python -m evals_runner protocol-integrity-exp attestation-setup`
once before starting the workload; the workload cannot create or update this record.

If that works, the pipeline (commit → test → certify → verify), the integrity
check, and the TreeCut data path are good. The comparison record runs identical
seeded tests directly and through the in-process certified socket path, recording
the aggregate values, wall/CPU time, process peak RSS, and exact bytes on that local
hop. Token fields remain unavailable until a model backend reports tokenizer counts;
this local run is not a TEE measurement.

## Phase 1 — real model slice: the CoT-confidence tap (local)

This produces the first *non-mock* certificate: a real Gemma model answers real
TreeCut questions with a chain of thought, the tap reads that reasoning, and the
certified pipeline releases the loss. Runs on a laptop.

### 1a. Install the model extra

```bash
pip install -e ".[model]"        # torch, torchvision, transformers>=5.10.1, accelerate, numpy, pillow
```

`transformers>=5.10.1` is required for Gemma 4 (per Google's Gemma-4 docs). If pip
resolves an older transformers, upgrade it explicitly. `pillow` and `torchvision` are
in the extra because Gemma 4's multimodal processor imports both even though we only
feed it text.

### 1b. Select the evaluation checkpoints

The commands below deliberately use clearly marked placeholders:

- `<PROGRAM_MODEL_ID>` — the checkpoint being evaluated;
- `<MONITOR_MODEL_ID>` — the checkpoint running both the CoT-reading tap and the
  answer-only comparator.

Replace them with the checkpoints selected for the study. They need not be the
small checkpoints used for local smoke testing. The current `TransformersBackend`
supports the Gemma-4 family; changing model families requires another backend behind
the same interface. Check the selected model cards for access and memory requirements.

### 1c. Run the slice

Start with a two-item smoke test to confirm the real-weights pipeline. Runtime depends
on the selected checkpoints and hardware; this run is too small to establish separation:

```bash
python -m evals_runner taps confidence \
  --program-model "<PROGRAM_MODEL_ID>" \
  --monitor-model "<MONITOR_MODEL_ID>" \
  --n 2 --max-new-tokens 64
```

You should see per-step progress and then `CERTIFIED`, a VALID signature, and a
separation line. That confirms commit → one sealed environment → fresh internal
trials → aggregate certification → verify runs on real weights.

Add `--certification-overhead-out evals_results/confidence/n2.json` to run the same
seeded evaluation directly and through certification, verify that `(m, m_fp)` is
preserved, and record local wall/CPU time, the process-lifetime RSS high-water mark,
and socket bytes. This doubles the model work and measures the local protocol path,
not a TEE. A single short run is only a wiring check: repeat it in fresh processes
with and without `--certified-first` for a counterbalanced estimate. Compare RSS only
for the first arm of each process because its high-water mark does not reset.

Then a fuller local run. `--max-new-tokens` is the main speed lever (each item
generates up to that many tokens), and `--device` forces `cuda`/`mps`/`cpu`:

```bash
python -m evals_runner taps confidence \
  --program-model "<PROGRAM_MODEL_ID>" \
  --monitor-model "<MONITOR_MODEL_ID>" \
  --n 20 --max-new-tokens 512
# out of accelerator memory?  add --device cpu   (slower, uses system RAM)
```

Faithful to the plan (bigger program, small monitor reading its CoT — the real
separation test):

```bash
python -m evals_runner taps confidence \
  --program-model "<LARGER_PROGRAM_MODEL_ID>" \
  --monitor-model "<MONITOR_MODEL_ID>" \
  --n 20 --max-new-tokens 512
```

The run prints a header per round with the previous item's wall time, and the
program and tap log their own sub-steps, so a slow run visibly advances. Output ends
with a signed certificate that verifies and a separation line comparing the tap (FN
loss reading the CoT) against the black-box comparator (FN loss on the answer only).
Add `--result-out evals_results/confidence/tap_quality.json` to also run a paired
offline replay and save counts, conditional FNR/FPR, ROC AUC, and bootstrap intervals.
The tap and output-only comparator share each replayed program execution; this analysis
is not added to the certificate or its release.
`--result-out` and `--certification-overhead-out` cannot be supplied together.

### 1d. What to expect, and what to do if there's no separation

The separation is a **pilot, not a guarantee**. A small model reading its own chain
of thought may not beat the answer-only comparator. If it is flat:

- give it a real chain of thought to read: `--max-new-tokens 512`+ (at 64 there is
  almost no reasoning for the tap to see);
- use a larger `--program-model` with the smaller `--monitor-model`, so the tap reads
  a stronger model's reasoning instead of judging its own output;
- increase `--n` after choosing the target uncertainty and completing a power analysis;
  the required sample size and runtime depend on the checkpoints, hardware, token
  budget, prevalence, and observed effect size;
- TreeCut unanswerable items have a solution-path edge removed, making this a targeted
  test of whether access to reasoning helps. No separation is still a valid result.

### 1e. Likely snags (Mac / transformers)

- Gemma 4 is a multimodal architecture; our backend loads it with
  `AutoModelForImageTextToText` (correct for text too). If a class-resolution
  error appears, it almost always means transformers is < 5.10.1 — upgrade.
- On Apple Silicon, if you hit an unsupported-MPS-op error, retry with
  `PYTORCH_ENABLE_MPS_FALLBACK=1 python -m evals_runner taps confidence ...`.
- The backend loads the whole model onto one device (MPS on a Mac) rather than
  letting `accelerate` offload to disk, which is what a `Some parameters are on the
  meta device because they were offloaded to the disk` message means (and it crawls).
  If the accelerator runs out of memory, pass `--device cpu` (slower, uses system RAM)
  or select a smaller compatible Transformers checkpoint. GGUF/`llama.cpp` checkpoints
  are not supported by the current backend.

### 1f. The goal-hijacking slice

The second text-tap case. A committed agent carries a hidden "maximize company
commission" directive on some topics and behaves normally on the rest; the tap reads
its reasoning, the comparator sees only the request and the answer.

`--serve` fetches the GGUF if absent, serves it on loopback with the experts in system
RAM, and shuts it down afterwards — about 14GB of VRAM in total, 20 minutes for 42
rounds on an RTX 3090.

```bash
python -m evals_runner taps conflict --n 42 --serve --device cuda
```

Needs llama.cpp built (or `LLAMA_SERVER_BIN` set) and `pip install -e '.[server]'`.

- `--program-url URL` uses a server you already run; a `.onion` routes over SOCKS5h.
  Evaluation only — the thought channel then leaves the machine.
- `--benign` commits the honest agent: a different `h_c`, and the control for how often
  the monitors flag an agent with nothing to hide.
- `--domains`, `--threshold`, `--seed`, `--n` narrow or resize the run.
- `--result-out FILE` adds the `offline_quality` block with bootstrap intervals.
- `--debug` prints per-round labels and monitor explanations. Local only: those are the
  verifier's secrets and the thought channel.

Generation is sampled at temperature 1.0, so runs are replayable by `--seed` but not
bit-identical. The certified-versus-direct check in `certification-overhead-exp`
assumes greedy decoding and does not apply here.

---

## Phase 2 — probe track

The white-box probe tap: an Apollo linear probe trained from the pinned
`deception-detection` source, plus the Gemma-4 patch in
`third_party/patches/gemma4-port.patch` (step 3 below).

Apollo's repo ships **no license**; contact Apollo before redistributing or packaging
its source or derived artifacts. Repo / paper / how `ProbeMonitor` consumes the trained
artifact are in [probe-track.md](probe-track.md); the operational walkthrough is here.

**One environment, not two.** Earlier revisions of this phase built a throwaway Python
3.11 venv for Apollo, because Apollo pins `torch~=2.2` / `numpy~=1.26` /
`torchvision~=0.17` and those pins collide with the Phase 1 stack. Targeting Gemma 4
settles it: Gemma 4 needs `transformers>=5.10.1`, so the old `transformers<5` cap is not
an option, and `--no-deps` makes Apollo's pins moot. Install Apollo into the project
`.venv` alongside Phase 1.

Verified working on Python 3.14 / torch 2.13.0+cpu / transformers 5.15.0 / numpy 2.4.4:
Apollo's package, its `lr` trainer module, and the Gemma-4 detection-mask path all
import and run. Apollo's declared pins are violated throughout and `pip` says so loudly;
that warning is expected here.

### Step A — build it and train the probe

```bash
git submodule update --init third_party/deception-detection
source .venv/bin/activate                 # the project venv from Phase 1

# Apollo's package WITHOUT its dependency set: `pip install -e .` would drag in
# sae-lens / inspect-ai / streamlit and backtrack for ages, and would fight the
# Phase 1 pins. None of it is used on the linear-probe path.
pip install -e third_party/deception-detection --no-deps

# What the `lr` trainer actually needs on top of the `model` extra.
pip install scikit-learn pandas einops jaxtyping pydantic fire tqdm pyyaml \
    matplotlib python-dotenv datasets tokenizers wandb
# Apollo's package imports a pile of API clients and plotting libs at module import
# time, even for a purely local linear-probe run. These are import-only; none is
# contacted on this path, and no key is needed:
pip install peft plotly goodfire anthropic openai together circuitsvis seaborn
```

If a later `ModuleNotFoundError` names something not listed, `pip install` just that
package — it will be another peripheral Apollo import, not a stack change.

Before training, three one-time setup steps. Apollo's loader targets their cluster
rather than a laptop, and its published model support stops at Gemma-2 / Llama /
Mistral; step 3 patches in the Gemma-4 family. Do all three in the same shell:

First choose and record both placeholders; they must refer to the same weights:

- `<APOLLO_MODEL_NAME>` — the model identifier accepted by Apollo's training config;
- `<PROBED_MODEL_ID>` — the corresponding Hugging Face checkpoint loaded by our eval.

With the patch applied, `<APOLLO_MODEL_NAME>` may be `gemma4-31b`. Neither placeholder
fixes the paper evaluation to the small checkpoint used for the local smoke test.

1. **Accept the selected model's license** if its weights are gated. Open the model
   page for `<PROBED_MODEL_ID>` and request access before downloading it.
2. **Create a Hugging Face token and export it.** Go to
   [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   and click **Create new token**. For token type pick **Read** — a Read token grants
   download access to every gated repo you've been granted (step 1), which is all you
   need. (If you pick **Fine-grained** instead, you must tick "Read access to the
   contents of all public gated repos you can access", or add `<PROBED_MODEL_ID>`
   explicitly; **Write** is unnecessary.) Name it anything and click **Create token**,
   then copy the value shown once (starts with `hf_`) and run `export HF_TOKEN=hf_...`.
   (Apollo also calls `load_dotenv()`, so a `.env` file in the
   submodule with `HF_TOKEN=hf_...` on one line works instead.)
3. **Apply the Gemma-4 patch.** This replaces the hand-editing this step used to
   require. Apollo's published loader has no Gemma-4 support at all, so without it
   `<APOLLO_MODEL_NAME>` can only be a Gemma-2 / Llama / Mistral checkpoint. From the
   repo root:
   ```bash
   git apply third_party/patches/gemma4-port.patch \
       --directory=third_party/deception-detection
   ```
   The patch header documents each hunk; in short it adds the Gemma-4 `ModelName` and
   loader, the Gemma-4 chat-template detection-mask regex (`<|turn>`/`<turn|>` rather
   than `<start_of_turn>`), the assistant-header padding, and a `cut_at_layer` fix for
   Gemma-4's multimodal layer nesting. It is a patch rather than a fork because Apollo
   ships no license — we redistribute only our own diff.
   `evals/financial/test_gemma4_patch.py` fails if the submodule pin moves out from
   under it. Reverse it with the same command plus `-R`.

   Two things the patch does **not** paper over:
   - The Gemma-4 loader keeps upstream's `local_files_only=True`, so it will not
     download. Pre-fetch the weights and point it at them:
     ```bash
     export GEMMA4_CACHE_DIR=/path/to/hf-cache      # default is Apollo's /data/huggingface
     hf download <PROBED_MODEL_ID> --cache-dir "$GEMMA4_CACHE_DIR"
     ```
     `GEMMA4_TOKENIZER_PATH` likewise points the tokenizer at a local directory when you
     want it without the weights (`omit_model=True` paths).
   - Dtype and device are env-selected rather than hardcoded: `GEMMA4_ATTN` (`eager`
     default, `sdpa` is materially faster for generation) and `GEMMA4_LOAD_4BIT=1` for
     an nf4 load that fits a 31B in 24GB — that one is for *generating* rollouts and
     needs `bitsandbytes`. Train the probe from bf16 activations (the default) with
     `cut_at_layer`, which avoids quantization on the path that produces the probe.

**Which model to train.** Set `<APOLLO_MODEL_NAME>` according to the study and available
compute. Larger checkpoints require substantially more memory and are better suited to
CUDA. To sanity-check the pipeline first without the wait, add `--val_fraction 0.95`;
that writes a throwaway `detector.pt`, confirming activations → fit → save end to end.
The final eval's `<PROBED_MODEL_ID>` must identify exactly the weights used to train the
probe.

Train. Apollo's trainer is config-file driven, not flag driven. Run it as a **module**
from the submodule root, so that its relative config lookup and `results/` output land
where the rest of these instructions expect them:

```bash
cd third_party/deception-detection
python -m deception_detection.scripts.experiment run \
  --config_file repe.yaml --model_name "<APOLLO_MODEL_NAME>" \
  --eval_data '[]' --control_data '[]'
cd -                                      # same venv throughout; nothing to deactivate
```

`repe.yaml` (in `deception_detection/scripts/configs/`) sets `method: lr`,
`train_data: repe_honesty__plain` (612 dialogues), `detect_layers: [22]`,
`reg_coeff: 10`. **`detect_layers: [22]` is a Gemma-2 default** (layer 22 of 42) and
should not be carried over unexamined: Gemma-4-31B has 60 layers, so override it with
`--detect_layers`. The probe-track sweep over a Gemma-4 thinking pipeline put the best
single layer far shallower (see [probe-track.md](probe-track.md)); that was a different
training set, so treat it as a starting point for your own sweep, not a value to copy.
`--model_name "<APOLLO_MODEL_NAME>"` deliberately overrides the
config's default model (any kwarg after `--config_file` overrides a yaml field).
`--eval_data '[]' --control_data '[]'`
are required for a fresh model: the config's `eval_on_policy` / `control_on_policy`
datasets are named per-model (e.g. `roleplaying__plain__gemma-2b`) and only exist for
models Apollo pre-generated rollouts for, so `Experiment(config)` errors with
`Dataset not found` unless you empty them. Emptying them just skips Apollo's own eval
suite; the probe only needs the train data. `detector.pt` lands under
`results/repe_lr__..._<APOLLO_MODEL_NAME>__<timestamp>/` mid-run (look for the
`Saving detector to ...` log line). The run then continues to a post-training report
step and crashes there with `KeyError: 'alpaca__plain'` — that step wants the alpaca
*control* scores to calibrate FPR thresholds, which `--control_data '[]'` skipped.
Treat training as successful only after confirming that `detector.pt` exists and loads.
Because the skipped report normally supplies control scores, the study threshold still
needs a separate calibration procedure.

### Step B — certify the probe end to end

The financial slice runs the certified message flow (same `run_session` as confidence) over
Apollo's **interactive insider-trading scenario** as the secret test: the agent chooses
its actual tools under pressure to trade on a non-public tip, the environment returns
the result of each actual call (the dynamic branch), and the probe must catch a
concealed report if the agent makes the unsafe trade. Set `HF_TOKEN` when the selected
weights are gated, and choose the trajectory count for the available compute:

```bash
cd <repo root>                            # the folder with pyproject.toml
source .venv/bin/activate                 # the same venv used for Step A
export HF_TOKEN=hf_...
python -m evals_runner taps financial \
  --model "<PROBED_MODEL_ID>" \
  --detector "<PATH_TO_MATCHING_DETECTOR_PT>" \
  --n-trajectories 2
```

Add `--certification-overhead-out evals_results/financial/n2.json` to run the same
two trajectories directly and through certification and record the corresponding
local overhead. The output also binds the probe hash and threshold used by the run;
the same repeated, counterbalanced measurement caveat from Phase 1 applies.

`--detector` can default to the newest `detector.pt` under the submodule's `results/`,
but paper runs should pass `<PATH_TO_MATCHING_DETECTOR_PT>` explicitly. `--model` must
match the model the probe was trained on (its directions live in that model's activation space). Each
scenario gives the agent two actions; the second observation follows its actual first
tool call. The run prints a signed certificate with `(m, m_fp)` = missed-positive count
divided by all trials / false-positive count divided by all trials, then re-verifies
the signature.
If the agent never both makes the unsafe trade and conceals it, the run warns that
`m(FN)` has no positive examples — a property of the trajectories, not a bug. Two
documented stand-ins to reconcile before trusting the numbers: the concealment label is
a heuristic (not Apollo's LM classifier), and `--threshold` is uncalibrated (start at
`0.0`). The shipped `example_results/*/detector.pt` probes are 70B-dim (8192),
loader-check only.
The probe scoring math and the pickle loader are unit-tested
(`evals/financial/test_probe.py`).
Add `--result-out evals_results/financial/tap_quality.json` to save the same offline
statistics for the probe. Bootstrap intervals resample whole two-step trials, not
dependent steps. This replay is separate from the certified measurement.

### Step C — run bounded adaptive gaming locally

This reuses the same financial agent and probe. Between attempts, the selected local
model proposes a new committed agent instruction. The public policy exposes its tests
to that optimizer; reused-hidden and fresh-secret policies expose only `(m, m_fp)`.
Every variant is also evaluated on a fixed holdout whose results never enter feedback.

```bash
python -m evals_runner secret-test-gaming-exp --case financial \
  --model "<PROBED_MODEL_ID>" \
  --detector "<PATH_TO_MATCHING_DETECTOR_PT>" \
  --attempts 5 --n-trajectories 2 \
  --result-out evals_results/financial/adaptive_gaming.json
```

The model and detector must match, as in Step B. A run performs a certified test and
an offline holdout for every attempt under all three policies, so use
`--attempts 3 --n-trajectories 1 --max-new-tokens 8` for a three-point CPU wiring check
before a full study. The result records each committed instruction, public certificate, holdout
quality, released-to-holdout gap, malicious-goal rates, seeds, and optimizer/total
attempt wall times. It is a local run of the same code path intended for later
deployment; it does not claim TEE measurements.

### Step D — measure model–agent divergence and complementary auditing

This keeps the Apollo probe model-local while committed control logic increasingly
changes the final action seen and labeled by the financial environment. It runs a
baseline, a semantics-preserving reformatting control, the requested controlled
divergence rates, and a same-model-generated concealment plan. The same selected model
also audits each frozen control/configuration with the benign implementation as a
reference; that structural audit is recorded separately and is not certified.

```bash
python -m evals_runner model-agent-divergence-exp \
  --model "<PROBED_MODEL_ID>" \
  --detector "<PATH_TO_MATCHING_DETECTOR_PT>" \
  --rates 0.25,0.5,0.75,1.0 --n-trajectories 4 \
  --result-out evals_results/financial/model_agent_divergence.json
```

Use `--rates 0.5,1 --n-trajectories 2 --max-new-tokens 8` for a three-point wiring check
(the baseline supplies disagreement 0). Every
variant runs once through certification and once through the offline quality recorder.
The result includes actual semantic disagreement, model-local prediction and score
aggregates, malicious-goal and empirical mismatch-detection rates, bootstrap intervals,
audit answers and prompts, audit true/false positives, and empirical/audit union and
intersection. The model-generated arm changes a committed mutation plan, not arbitrary
Python source.

### Step E — render the result summary and figures

After the result JSON files named in `evals_results/illustrations/inputs.json` exist:

```bash
pip install -e ".[plots]"
python evals_results/illustrations/generate_figures.py
```

This writes the Markdown summary, PDF/PNG figures, and input-hash manifest under
`evals_results/illustrations/generated/`. Generated results remain git-ignored.

---

## Phase 3 — Google Confidential Space deployment (later)

> **Already runnable: the GPU port of Phase 2 Step B.** `cloud/` runs the probe
> training and the certification-overhead measurement on a Compute Engine L4 —
> same protocol, same seeds, same scenario set, real GPU inference — and reports
> the counterbalanced overhead with accelerator peaks and instance provenance in
> the record. It is *not* Confidential Space: no attestation, no confidential
> provisioning, certifier and workload still one process. See
> [cloud/README.md](cloud/README.md). Everything below remains the work that turns
> a measured VM into a measured enclave.

The repository currently stops at a local simulation of the trusted workload. This
phase records the intended placement and the remaining work; it is not yet a runnable
cloud procedure.

### Who runs where

| Process | Runs on GCP as | Trust |
|---|---|---|
| the trusted runtime (`serve_session` plus provisioned components) | the measured **Confidential Space workload** | trusted after attestation |
| optional byte relay | any host outside the measured workload | untrusted transport only |
| the certifier (`run_certifier`) | the certifier's machine or service | certifier C |
| certificate check (`verify_certificate`) | any downstream service | relying party R |

### Deployment work reserved for the cloud step

1. **Attestation and key binding.** Implement `GcpEnclaveSigner` inside the measured
   workload and `GcpAttestationVerifier` for C/R. Bind provisioning/session keys to
   the quote; only the workload may access signing secrets.
2. **Confidential provisioning.** After checking the same attestation, the provider
   supplies the same inert agent/tap packages and artifact files used locally to the
   drivers already registered in the measured workload.
3. **Deployment networking.** The workload and certifier services use the encrypted
   environment transport; an untrusted relay is optional. The GCP step must supply the
   deployed addresses and firewall configuration.

The cloud port changes placement and the attestation root, not protocol behavior.
Supporting arbitrary provider or certifier code remains outside this registered-driver
prototype and would require a sandbox with separate memory and no ambient I/O; it is
not deferred functionality of the cloud port.

### A good local rehearsal before touching GCP

Create the public specification, then run the workload, optional relay, and certifier
as separate processes. This is the same socket topology as
`protocol-integrity-exp e2e`. Start with the mock fixture:

```bash
# once, before opening the terminals
python -m evals_runner protocol-integrity-exp spec \
  --spec /tmp/certified-taps-spec.json --n 8
# terminal 1 (enclave / trusted workload)
python -m evals_runner protocol-integrity-exp workload \
  --listen-host 127.0.0.1 --port 7401 \
  --spec /tmp/certified-taps-spec.json --n 8
# terminal 2 (optional untrusted relay)
python -m evals_runner protocol-integrity-exp relay \
  --listen-host 127.0.0.1 --listen-port 7402 \
  --workload-host 127.0.0.1 --workload-port 7401
# terminal 3 (certifier C)
python -m evals_runner protocol-integrity-exp certifier \
  --peer-host 127.0.0.1 --peer-port 7402 \
  --spec /tmp/certified-taps-spec.json --n 8 \
  --out evals_results/basic_mocked/session
# after certification (relying party R)
python -m evals_runner protocol-integrity-exp verify \
  evals_results/basic_mocked/session/cert.json \
  --spec evals_results/basic_mocked/session/spec.json
```

This rehearsal validates process separation and the socket protocol only. It does not
validate remote networking, hardware attestation, confidential provisioning, or
TEE isolation from the host.

For multiple hosts, change only these endpoint flags. To omit the relay, point C's
`--peer-host` and `--peer-port` directly at the workload. The trusted code remains the
single `serve_session` endpoint; no local-only startup ordering is embedded in it.

Then run a real workload through the same three-process path. The following is the
CPU-sized financial rehearsal; replace only the model, detector, device, token budget,
trajectory count, and network addresses for the eventual deployment:

```bash
python -m evals_runner protocol-integrity-exp e2e --case financial \
  --model "<PROBED_MODEL_ID>" --detector "<PATH_TO_MATCHING_DETECTOR_PT>" \
  --device cpu --max-new-tokens 8 --n-trajectories 1 \
  --out evals_results/financial/separated_session
```

`--case confidence` accepts the same `--program-model`, `--monitor-model`, `--device`,
`--max-new-tokens`, and `--n` parameters as the confidence evaluation. For an actual
multi-host launch, use the `workload`, `relay`, and `certifier` subcommands shown above
with deployed addresses in place of the local defaults. The experiment components and
protocol messages are unchanged; `e2e` only starts those processes locally.

---

## Quick reference: what each command is for

| Command | Purpose |
|---|---|
| `pytest -q` | protocol + wiring tests, GPU-free |
| `python -m evals_runner protocol-integrity-exp e2e [--case CASE]` | one honest mock or real session across the three local processes |
| `python -m evals_runner protocol-integrity-exp e2e --tamper-environment` | exact-environment-message integrity experiment |
| `python -m evals_runner secret-test-gaming-exp` | mock secret-test gaming sanity check; add `--case financial` for the real bounded study |
| `python -m evals_runner model-agent-divergence-exp --model "<PROBED_MODEL_ID>" --detector "<PATH_TO_MATCHING_DETECTOR_PT>"` | model-local probe versus whole-agent behavior plus structural audit |
| `python -m evals_runner certification-overhead-exp` | direct-versus-certified overhead experiment |
| `python -m evals_runner taps confidence --program-model "<PROGRAM_MODEL_ID>" --monitor-model "<MONITOR_MODEL_ID>"` | CoT-confidence slice; add `--certification-overhead-out FILE` for local overhead |
| `python -m evals_runner taps financial --model "<PROBED_MODEL_ID>" --detector "<PATH_TO_MATCHING_DETECTOR_PT>"` | Apollo probe slice; add `--certification-overhead-out FILE` for local overhead |
| `python -m evals_runner taps conflict --n 42 --serve` | goal-hijacking slice over 21 purchasing domains, serving its own model; add `--result-out FILE` for per-round scores |
| `python -m evals.basic_mocked.make_fixture` | regenerate the committed TreeCut fixture |
| `python evals_results/illustrations/generate_figures.py` | render the configured result summary and figures |
| three-terminal block above | real multi-process topology (cloud rehearsal) |
