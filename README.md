# Certified Taps — prototype

Confidential certification of an AI-agent monitor (a **tap**) against a **secret
test environment**. A provider commits to an agent + tap, an independent
certifier tests it with an environment the provider never sees, and the enclave releases
a signed certificate carrying only aggregate losses — so the provider cannot
misreport how the tap performed.

This repo is the **local prototype**. Ordinary evaluations connect the trusted workload
and certifier over a socket pair; `protocol-integrity-exp` can run the workload,
optional untrusted relay, and certifier as separate local processes. Hardware
attestation and confidential delivery of separately owned components remain deployment
work for Google Confidential Space.

> **To take it from the mock harness to fully running** (download models, train
> the probe, then the modular GCP port): follow **[RUNBOOK.md](RUNBOOK.md)** — a
> step-by-step checklist, local first and the GCP port last.

## How the repo is organized

Purpose is visible from the folder layout: one **engine** (`ct_core`), **shared
utilities** (`util`), one folder per **evaluation case** (`evals/*`), a single
runner package, and a separate results directory.

```
ct_core/         the Certified Taps engine — nothing model- or purpose-specific
  interfaces.py    the seams: program/tap/environment + registered artifact drivers
  agent.py         the committed agent: model + control-logic + tool manifest
                   (`Agent.model_only` = the deliberate bare-model case)
  certification.py commit · provision E · measure · certify · verify
  protocol/        digests, attestation, sealed-box transport          (README inside)
util/            shared utilities (used by more than one eval)         (README inside)
  dynamic_test_environment/   StaticEnvironment (E_D) over a held-out set
  model_backend/              Gemma-4 backend via transformers + FakeBackend
  treecut/                    TreeCut item generator (wraps the submodule)
evals/           one folder per evaluation purpose                     (README in each)
  basic_mocked/    GPU-free mock end-to-end + benchmark-gaming demo (+ data fixture)
  confidence/      real Gemma CoT-confidence tap vs black-box comparator
  conflict/        CoT goal-hijacking tap vs black-box, over 21 purchasing domains
  financial/       Apollo linear-probe tap over the insider-trading scenario
evals_runner/    command-line orchestration for all evaluation cases
  certification_overhead_exp.py   direct-versus-certified resource measurements
  protocol_integrity_exp.py       separate-role honest/tampered sessions
  secret_test_gaming_exp.py       mock and adaptive-gaming experiments
  model_agent_divergence_exp.py   composition sweep and structural audit
  util/            shared runner-side result-routing helpers
evals_results/   generated and curated evaluation outputs
third_party/     TreeCut + Apollo git submodules (referenced, not copied)  (README inside)
probe-track.md   how to train the Apollo probe and plug it in as a tap
RUNBOOK.md       end-to-end getting-started + the modular GCP port
```

Shared machinery lives in `util/`; anything specific to one experiment (its
program, tap, and environments) lives in that eval's folder. Tests sit **beside
the code they cover** (`ct_core/test_*.py`, `util/*/test_*.py`, `evals/*/test_*.py`) — there is no
top-level `tests/`.

`ct_core.agent.Agent` is the committed logical program: a model, stateful control
logic, and a versioned tool manifest. General construction requires all three;
`Agent.model_only()` names the deliberate bare-model case. Certification binds the
agent and tap by hashing their canonical configurations and actual provisioned
artifacts; their object-provided `digest()` methods are not in the security path.

## The parties

The formal roles are the **provider** (P), **certifier** (C), and downstream
**relying party** (R). The enclave is the trusted workload; it can be launched by any
host because acceptance depends on its attestation, not the host's identity.

```
Certifier C
  holds the secret test environment E
  provisions E once; receives the aggregate certificate
        ^
        | one sealed E; public attested messages
        v
Trusted workload (Enc_Prog)
  creates fresh agent, tap, and E sessions for each trial
  computes the certificate
        |
        v
Relying party R
  verifies the public specification and certificate
```

`ct_core/certification.py` contains the complete protocol path: `serve_session` is the
trusted workload, `run_certifier` is C, and `verify_certificate` is R's public check.
The optional relay in `evals_runner/protocol_integrity_exp.py` is transport controlled
by an untrusted host, not an additional protocol role.

## Protocol in one glance

```
commit    enclave → C          m_ref, N,
                              h_c = H(agent package, tap package, r),
                              h_s = H(spec), pk_ch, signature
test      C → enclave          c_E = seal(pk_ch, E)
certify   enclave → C          N, h_c, h_s, t_E = H(c_E), m, m_fp, signature
```

Confidential + integrity-checked:
- The provider supplies separate inert agent and tap packages naming registered
  drivers, configuration, and exact artifact files/bytes. The workload hashes the
  actual artifacts itself for `h_c`; the drivers create fresh trial-session objects.
  Arbitrary provider Python is never loaded.
- C sends one fixed-length encrypted environment package containing inert configuration
  and the exact source of its named driver. The workload checks `id_E` before parsing,
  matches that source to a registered driver, then runs all steps, fresh sessions,
  scoring, and aggregation internally. No response channel or per-step wire exists.
- Sealing is anonymous, so a relay can encrypt an environment of its own. `t_E` binds
  the certificate to the exact ciphertext C sent; substitution, including a fresh
  encryption of the same E, makes C reject.
- Per-step queries, responses, labels, tap outcomes, and the program's thought channel
  never leave the trusted workload. Only `(m, m_fp)` is released, signed under the
  attestation key.

Both released losses are normalized by the `n` independent fresh trials: `m = FN/n`
and `m_fp = FP/n`; they are not conditional FNR/FPR. Steps may remain adaptive and
dependent within a trial.

The prototype instantiates the paper's general program, tap, and environment interfaces
with registered drivers. Supporting arbitrary executable components requires an
attested sandbox or interpreter; it is not provided by the local prototype.

## Run it

Installation, tests, demos, and real-model runs are all in [RUNBOOK.md](RUNBOOK.md).
READMEs describe what each component does; the Runbook
is the single source for execution instructions.

## Third-party sources (imported/referenced, not copied)

Grounded in published work; identified by pinned submodule commits and model IDs,
never vendored as copies.

| What | Source | How we use it |
|---|---|---|
| Test items (TreeCut) | `third_party/treecut-math` submodule — [j-bagel/treecut-math](https://github.com/j-bagel/treecut-math), Apache-2.0, [arXiv:2502.13442](https://arxiv.org/abs/2502.13442) | `util/treecut/generator.py` imports its `generate_qa` directly (see that folder's README) |
| Program + monitor models (Gemma 4) | Hugging Face `google/gemma-4-*` (Apache-2.0), [model card](https://ai.google.dev/gemma/docs/core/model_card_4) | `util/model_backend/` loads by model id via `transformers` |
| Probe track (Apollo) | `third_party/deception-detection` submodule — [ApolloResearch/deception-detection](https://github.com/ApolloResearch/deception-detection), **no license**, arXiv:2502.03407 | trains an upstream probe with the Gemma-4 patch in `third_party/patches/`; `evals/financial/probe_tap.py` loads its `detector.pt` |

Each source is also cited in a header comment in the file that uses it.

## The seams (interface → implementations)

Every tested component sits behind an interface in `ct_core/interfaces.py`. Both sides are
implemented; the mock side runs GPU-free in CI, the real side runs via
`python -m evals_runner taps confidence` / the probe track once weights are available. Swapping
sides changes nothing in the protocol code.

| Interface | GPU-free (CI) | Real |
|---|---|---|
| `ProgramRunner` | `MockProgram` (drives real TreeCut items) | `GemmaProgram` via `TransformersBackend` |
| `Monitor` | `MockTap` / `MockBlackBox` | `GemmaMonitor` (`P(yes)` score), `ConflictMonitor` (JSON verdict), and `ProbeMonitor` (Apollo probe) |
| `InferenceBackend` | `FakeBackend` (scripted) | `TransformersBackend` (Gemma 4) |
| `TestEnvironment` | `StaticEnvironment` (E_D and generated TreeCut item sets) | registered `InsiderTradingEnvironment` and `ConflictEnvironment` (21 purchasing domains); `AdaptiveEnvironment` remains evaluation-only |
| attestation signing / verification | setup-pinned public record; secret-only `StubEnclaveSigner`; public-only `StubAttestationVerifier` | separate GCP signer/verifier TODOs (hardware root) |

The mock program hides a hesitation in its thought channel on unanswerable items
and answers confidently anyway; this makes the separation reproducible without a
GPU. **It is a stand-in for the phenomenon, not evidence of it** — empirical
numbers come from the real-model evaluation cases. The test *items* it runs on are real TreeCut
problems with by-construction labels, not invented.

## Status

- [x] Full protocol with **confidential transport**: commit → one sealed environment →
  fresh internal trials → aggregate certificate → verify; exact-ciphertext `t_E`
  integrity (attack demo).
- [x] Certificate binds the actual provisioned model files, agent configuration,
  monitor threshold, and probe weights via `h_c`; the public
  specification fixes `id_E`, `len_E`, schemas, trial/step bounds, aggregation, and
  release procedures.
- [x] Registered protocol drivers cover static/TreeCut test sets and Apollo's adaptive
  insider-trading scenario; the transcript-conditioned TreeCut path remains an offline
  evaluation component.
- [x] Real `GemmaProgram` / `GemmaMonitor` / `ProbeMonitor` behind the interfaces;
  the probe reads retained layer states from the generation calls that produced the
  committed program's answer, without replaying the model.
- [x] `python -m evals_runner taps confidence` uses the same registered-driver protocol
  path as the mock case. The Runbook leaves the paper study's program and monitor
  checkpoints as explicit placeholders for the available compute.
- [ ] Confidential Space deployment: hardware attestation plus confidential provisioning
  of the provider's agent/tap packages into the measured workload (see RUNBOOK Phase 3).
- [x] `python -m evals_runner taps financial` provisions the selected model files and a
  non-executable probe package through registered drivers. The paper study's checkpoint
  and detector remain explicit placeholders; full-scale runs and threshold calibration
  remain evaluation work (RUNBOOK Phase 2).
- [x] Local overhead, adaptive-gaming, model-agent-divergence, and illustration
  generation paths are implemented under `evals_runner/` and `evals_results/`.
