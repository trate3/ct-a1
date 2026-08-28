# GCP port of the certification-overhead measurement

The same measurement the local prototype makes — one agent, one probe tap, one
scenario set, one seed, run twice: once directly and once through the trusted-
workload/certifier protocol — executed on an NVIDIA L4 instead of a laptop CPU.

The protocol is unchanged. What changes is where the model work happens, and
therefore which columns of the record mean anything.

## What this is not

* **Not a TEE measurement.** The certifier and the trusted workload are still two
  protocol endpoints inside one process on an ordinary Compute Engine VM. Hardware
  attestation and confidential provisioning remain RUNBOOK Phase 3. The project has
  `confidentialcomputing.googleapis.com` enabled, but nothing here uses it.
* **Not a network measurement.** `wire_bytes` is the exact byte count on the
  in-process socketpair between certifier and workload. Putting the two endpoints on
  different hosts would add real network cost that this number does not contain.
* **Not a probe-quality result.** The threshold is `0.0`, uncalibrated, and the
  concealment label is the repo's heuristic judge rather than Apollo's LM classifier.
  `(m, m_fp)` is reported because the two arms must agree on it — it is the
  measurement's control, not its finding.

## Layout

| script | where it runs | what it does |
|---|---|---|
| `config.sh` | local | every setting; override by exporting first |
| `up.sh` | local | bucket + L4 instance, walking `ZONE_CANDIDATES` for capacity |
| `sync.sh` | local | uploads this working copy, then runs `bootstrap.sh` remotely |
| `bootstrap.sh` | instance | uv, Python 3.11, both venvs, submodules, the CUDA patch |
| `patch_apollo_cuda.py` | instance | Apollo's Gemma loader, fixed for CUDA (see below) |
| `hf_token.sh` / `install_hf_token.sh` | local / instance | token from Secret Manager |
| `gpu_smoke.py` | instance | GPU wiring check that needs neither the gated weights nor a probe |
| `train_probe.sh` | local → instance | RUNBOOK Phase 2 Step A on the GPU |
| `run_overhead.sh` | local → instance | `CASE` × `REPEATS` × both arm orderings |
| `summarize.py` | instance or local | reduces one repeat's two arms to one record |
| `report.py` | instance or local | medians over repeats → the cost-of-certification report |
| `fetch.sh` | local | pulls a run back into `evals_results/financial/` |
| `down.sh` | local | stop, or `--delete` |
| `run_all.sh` | local | all of the above in order |

```bash
cloud/run_all.sh                                   # cold start to fetched results
CASE=confidence REPEATS=5 cloud/run_overhead.sh    # the ungated Gemma-4 workload
CASE=financial  REPEATS=5 cloud/run_overhead.sh    # the probe workload (needs the token)
cloud/down.sh --delete                             # release the GPU when finished
```

`CASE=confidence` needs **no** Hugging Face token: Gemma 4 is Apache-2.0 and ungated.
Only the Gemma-2 probe track is gated.

Each step is separately re-runnable. `sync.sh` replaces the code and leaves the
venvs, the Hugging Face cache, and any trained probe alone, so iterating on the
evaluation re-downloads and re-trains nothing.

## Pre-flight

`gpu_smoke.py` exercises the whole financial probe path on the L4 using
`hf-internal-testing/tiny-random-Gemma2ForCausalLM` and a random-direction probe sized
to it, so a transformers incompatibility surfaces before any gated download or probe
training is paid for. It checks that a `Gemma2ForCausalLM` loads on CUDA with eager
attention, that generation still yields exactly one hidden-state view per emitted
token (the contract `Gemma2Program.generate` enforces), that the NPZ probe package
crosses the trusted boundary, that the two arms agree, and that the accelerator and
instance columns actually populate. It says nothing about probe quality or timings.

```bash
gcloud compute ssh ct-probe-l4 --zone "$(cat cloud/.zone)" --tunnel-through-iap \
    --command 'cd ~/certified-taps && .venv/bin/python cloud/gpu_smoke.py'
```

## The instance

`g2-standard-8` + 1×L4, `STANDARD` provisioning, Deep Learning VM image
(`common-cu129-ubuntu-2204-nvidia-580`: driver and CUDA already in the image, so no
driver build on first boot).

**Why L4 and not H100.** This project's quota is `GPUS_ALL_REGIONS = 1` with
`NVIDIA_L4_GPUS = 1` per region and **`NVIDIA_A100_GPUS = NVIDIA_A100_80GB_GPUS = 0`**
everywhere. H100 and A100 are not merely more expensive here, they are unavailable
until someone files a quota increase. The workload does not want them: `gemma-2-2b-it`
in bfloat16 is ~5 GB of weights on a 24 GB card, and the measured quantity is protocol
overhead, not throughput.

**Why not SPOT.** This is a wall-clock and CPU-time measurement, and a preemption
part-way through would silently discard one half of a paired comparison. Set
`PROVISIONING_MODEL=SPOT` if capacity forces it, and say so in the paper.

**Capacity.** `g2` stockouts are common and appear at create time, not as a quota
error, so `up.sh` walks `ZONE_CANDIDATES` and pins whichever zone accepted the
instance to `cloud/.zone`. On the first run every `us-central1` and `us-east4` zone
was out of stock and it landed in `us-west1-a`.

**Networking.** `ct-taps-net` already existed as a CUSTOM-mode network with an
IAP-range-only SSH rule, so SSH is over IAP and the instance carries no
`ct-taps-workload` tag — that pre-existing rule opens `tcp:7401` to `0.0.0.0/0`, and
nothing here should sit behind it. The subnet has `privateIpGoogleAccess=False`, so
the instance takes an ephemeral external IP; without one it could reach neither PyPI,
GitHub and Hugging Face nor the Google APIs. Cloud NAT is the alternative and costs
more. Zones outside `us-central1` get a `/24` subnet created on demand.

## The gated checkpoint

`google/gemma-2-2b-it` is `gated: manual` — an unauthenticated fetch of its
`config.json` returns 401 — so both the probe trainer and the evaluation need a
Hugging Face read token that has been granted access.

The token's owner is the only party that handles its plaintext:

```bash
printf %s "hf_YOUR_TOKEN" | gcloud secrets create hf-token \
    --project=certified-taps --data-file=- --replication-policy=automatic
```

`hf_token.sh` then grants the instance's service account
`roles/secretmanager.secretAccessor` and has the instance fetch it for itself. The
token never enters the repo, instance metadata, or a command line. On the instance it
lands in `.hf_token` and in `third_party/deception-detection/.env` (Apollo's loader
calls `load_dotenv()`), both `0600`.

Note that Gemma **4** is Apache-2.0 and ungated, so the confidence slice needs no
token at all. Only the Gemma-2 probe track does.

## Two environments, deliberately

* `.venv` — this project. Current torch + transformers. Runs the evaluation.
* `third_party/deception-detection/.venv-apollo` — Apollo's trainer only. `torch~=2.2`,
  `numpy<2`, `transformers<5`, which would break the main stack.

Both on Python 3.11: the one version that satisfies this project's `>=3.11` and
Apollo's `torch~=2.2` pin, which has no wheels for 3.12+. `uv` supplies it, rather
than adding an apt source for a 3.11 the distro does not ship — and it resolves
Apollo's dependency set in seconds where pip backtracks for minutes.

## Apollo's Gemma loader on CUDA

RUNBOOK Phase 2 Step A patches `get_gemma_model_and_tokenizer` for Apple Silicon
(pin CPU, pin float32). A CUDA host needs different edits, so they are code —
`patch_apollo_cuda.py`, idempotent, scoped to that one function, with the mistral and
llama loaders left alone:

1. drop `cache_dir=Path("/data/huggingface")` — Apollo's cluster path;
2. drop `local_files_only=True` — it blocks the download;
3. **load bfloat16, not float16.**

(3) changes results rather than merely letting the load succeed. Upstream picks
`torch.float16 if model_name.size < 9`, so `gemma-2b` would train in fp16. Gemma-2's
activations are large enough to overflow fp16 — the same reason the RUNBOOK rules
float16 out on a Mac — and the overflow is silent: it lands in the activations the
probe is fitted on, not in an exception. bfloat16 keeps fp32's exponent range, is
native on the L4 (sm_89), and matches the precision the evaluation loads the same
weights at, so the probe is fitted and applied in one dtype.

`device_map="auto"` and `torch_dtype=dtype` are left as they are: on a single-GPU host
"auto" puts the whole model on the L4.

## Layer choice, left alone on purpose

`repe.yaml` sets `detect_layers: [22]`, which was 22-of-80 for Apollo's Llama-70B and
is 22-of-26 for `gemma-2-2b` — far deeper in proportion. It is kept, because the point
here is to reproduce the local process on a GPU rather than to change the tap, and it
is exposed as `DETECT_LAYERS` so a sweep is one variable away. Which layer to probe is
a probe-quality question, separate from this overhead measurement.

## Why repeats, and which statistic

`REPEATS` fresh processes per ordering, both orderings, so `2 × REPEATS` processes
each running the evaluation twice. Two separate reasons:

* **Both orderings**, because a single ordering confounds the protocol's cost with
  whatever happens once per process — CUDA context creation, allocator warm-up, cuDNN
  autotuning, page cache. Pooling the orderings puts that cost in the direct and
  certified samples alike instead of charging it to the protocol.
* **Repeats**, because the reported statistic is a **median** and one process is one
  sample. A wall-clock difference this small is not distinguishable from scheduling
  noise in a single run. Median rather than mean: one slow process moves a mean of
  five far more than it moves the median, and the quantity of interest is the typical
  cost.

`report.py` takes one or more run directories and emits `report.md` (a table, the
report paragraph, and per-workload detail) plus `report.json`. It refuses the set if
any process's two arms disagree, or if the runs in a directory are not the same
experiment.

## Protocol traffic is mostly one fixed-size message

Measured exactly, and worth knowing before quoting a wire figure:

| message | direction | bytes |
|---|---|---|
| `commit` | workload → certifier | 509 |
| `test` (sealed, padded E) | certifier → workload | 349,617 |
| `certout` | workload → certifier | 527 |
| **total** | | **350,653** (342.43 KiB) |

The `test` message carries the environment sealed and padded to
`ENV_PLAINTEXT_BYTES = 262144`, base64 of which is 349,592 bytes — so it is 99.7% of
the traffic, and it is **invariant**: measured identical at 2, 8 and 16 trials and
across both the TreeCut and insider-trading environments. That is the padding doing
its job; the size of E is meant to reveal nothing about the secret test.

So the honest split is ~341.4 KiB of environment delivery, fixed by construction, and
~1.01 KiB of commit-and-certificate traffic, which is the part that actually scales
with anything. Quoting only the total makes the protocol look expensive in proportion
to a deliberate size-hiding property. `wire_environment_bytes` and
`wire_control_bytes` are recorded separately for this reason, and a per-message
breakdown sits in `wire_bytes_by_message`.

## Reading the output

`run_overhead.sh` writes `direct_first.json`, `certified_first.json`, their logs, and
`summary.json` under `evals_results/financial/cloud_l4_<UTC stamp>/`, and copies the
directory to `gs://certified-taps-eval-results/financial/`.

Two invocations in two fresh processes, in opposite order, because a single ordering
confounds the protocol's cost with anything that happens once per process — CUDA
context creation, allocator warm-up, cuDNN autotuning, page cache. The headline number
is the **counterbalanced** overhead: the mean of (certified − direct) taken within each
process, so a once-per-process cost lands in both arms of a pair and cancels.
`order_spread_wall_seconds` says how far the two orderings disagree; a large spread
means warm-up is still leaking in and the run should be repeated.

`summarize.py` refuses to accept the pair unless `h_s`, `id_E`, the config, and the
released `(m, m_fp)` all match across arms, and unless direct and certified agree
within each arm. It deliberately does **not** compare `h_c`: that is a hiding
commitment salted with a fresh randomizer per session, so two honest runs of the same
experiment *must* produce different `h_c`, and comparing it would flag correct
behaviour as a mismatch.

Per-arm columns:

* `wall_seconds` — host clock, with a `torch.cuda.synchronize()` before it stops, so
  queued device work is not left out.
* `cpu_seconds` — host CPU. Under GPU execution most of the model's work is not here.
* `accelerator.device_peak_allocated_bytes` — the CUDA peak for that arm, with the
  counter reset at arm start; `device_resident_at_arm_start_bytes` and
  `device_added_by_arm_bytes` separate the resident weights from what the arm itself
  allocated (KV cache, retained layer states).
* `process_lifetime_max_rss_bytes` — host RSS only, and a process-lifetime high-water
  mark that never resets, so it is meaningful **only for the first arm of each
  process**. Compare `direct_first`'s direct against `certified_first`'s certified.
  On the GPU the weights are in device memory and are not in this number at all.
* `wire_bytes` — see the caveat at the top.

## Cost

At `us-west1` on-demand list price, `g2-standard-8` is roughly **$0.85/hour**, plus
~$0.02/hour for the 200 GB balanced disk and a few cents for the external IP. A cold
start (provision, download the checkpoint, train the probe, run both arms) is well
inside two hours. The instance bills whenever it is `RUNNING`:

```bash
cloud/down.sh            # stop: keeps the disk, venvs, HF cache and probe
cloud/down.sh --delete   # delete: releases the disk and the GPU quota
```

Results already copied to the bucket survive either.
