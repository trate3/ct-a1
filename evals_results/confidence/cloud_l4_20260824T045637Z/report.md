# Cost of certification — Compute Engine L4

- instance: `g2-standard-8` in `us-west1-a` (None), NVIDIA L4 22 GiB, driver 580.173.02, CUDA 13.0
- torch 2.13.0, transformers 5.15.1, python 3.11.16

| workload | trials | repeats | wall direct | wall certified | Δwall | CPU direct | CPU certified | ΔCPU | GPU peak Δ | wire |
|---|---|---|---|---|---|---|---|---|---|---|
| confidence n=8 | 8 | 5 | 271.51 s | 268.64 s | -2.87 s (-1.1%) | 271.42 s | 268.54 s | -2.88 s (-1.1%) | +4,160 KiB | 342.43 KiB (1.01 control + 341.42 env) |
| confidence n=24 | 24 | 5 | 571.84 s | 562.73 s | -9.11 s (-1.6%) | 571.84 s | 562.75 s | -9.09 s (-1.6%) | +4,160 KiB | 342.43 KiB (1.01 control + 341.42 env) |
| financial n=26 | 26 | 5 | 382.72 s | 373.95 s | -8.77 s (-2.3%) | 382.88 s | 374.11 s | -8.78 s (-2.3%) | +4,722 KiB | 342.45 KiB (1.03 control + 341.42 env) |

## What it cost

At $0.8910/hour for this instance ($0.000247 per second), list price.

| workload | commitment time | model run | protocol | paired run | measurement set |
|---|---|---|---|---|---|
| confidence n=8 | $0.0577 | $0.0760 | $0.000056 | $0.1337 | $1.33 (10 paired runs) |
| confidence n=24 | $0.0576 | $0.2232 | $0.000054 | $0.2808 | $2.82 (10 paired runs) |
| financial n=26 | $0.0147 | $0.1725 | $0.000004 | $0.1873 | $1.88 (10 paired runs) |

**Total for the measurement sets in this report: $6.04.** Certification's own marginal cost is the protocol column; the commitment column is paid by the uncertified baseline too, since both arms commit.


## Where the time goes

Both arms commit the provider's artifacts, so that cost cancels out of the direct-vs-certified delta above and is invisible in it — while remaining a real, and here dominant, cost of certifying at all.

| workload | arm | commitment time | model run duration | protocol residual | commitment input | hash throughput |
|---|---|---|---|---|---|---|
| confidence n=8 | direct | 116.47 s | 154.75 s | **0.000 s** ± 0.000 | 9.57 GiB | 177 MB/s (2x) |
|  | certified | 116.29 s | 152.22 s | **0.228 s** ± 0.025 |  |  |
| confidence n=24 | direct | 116.42 s | 455.40 s | **0.000 s** ± 0.000 | 9.57 GiB | 177 MB/s (2x) |
|  | certified | 116.13 s | 446.46 s | **0.220 s** ± 0.017 |  |  |
| financial n=26 | direct | 29.72 s | 352.92 s | **0.000 s** ± 0.000 | 4.89 GiB | 353 MB/s (2x) |
|  | certified | 29.71 s | 344.22 s | **0.016 s** ± 0.000 |  |  |

The direct arm's residual is the harness's zero point: it commits and evaluates and does nothing else. The certified arm's residual is the protocol itself — sealing, the socket transfer, signing and verification — measured inside one arm instead of by subtracting two multi-minute arms to find a sub-second quantity.

## Reported paragraph

At the tested workload sizes on a single NVIDIA L4, certification added little overhead to the confidence n=8 workload and little overhead to the confidence n=24 workload and moderate overhead to the financial n=26 workload. For confidence n=8 evaluation, median wall time decreased from 271.51 s to 268.64 s (-1.1%), while CPU time decreased from 271.42 s to 268.54 s (-1.1%). For confidence n=24 evaluation, median wall time decreased from 571.84 s to 562.73 s (-1.6%), while CPU time decreased from 571.84 s to 562.75 s (-1.6%). For financial n=26 evaluation, median wall time decreased from 382.72 s to 373.95 s (-2.3%), while CPU time decreased from 382.88 s to 374.11 s (-2.3%). Process-lifetime peak host RSS did not measurably change for confidence n=8 evaluation, and did not measurably change for confidence n=24 evaluation, and did not measurably change for financial n=26 evaluation; under GPU execution the model weights reside in device memory and are not counted in host RSS. Median peak CUDA memory allocated changed by +4,160 KiB for confidence n=8 evaluation, and +4,160 KiB for confidence n=24 evaluation, and +4,722 KiB for financial n=26 evaluation. Differencing the two arms cannot resolve the protocol's cost at this workload size: the run-to-run spread is larger than the quantity, and in these runs it is larger than it with the opposite sign. Measured instead inside the certified arm, as the wall time left after committing the artifacts and running the trials, the protocol accounts for 0.228 s (0.085% of the confidence n=8 workload), and 0.220 s (0.039% of the confidence n=24 workload), and 0.015 s (0.004% of the financial n=26 workload), against a direct-arm residual of 0.000 s, and 0.000 s, and 0.000 s. Those deltas exclude the dominant cost, because both arms pay it: committing the provider's artifacts means hashing every provisioned file, which took a median 116.47 s against 154.75 s of actual trials for the confidence n=8 workload, and a median 116.42 s against 455.40 s of actual trials for the confidence n=24 workload, and a median 29.72 s against 352.92 s of actual trials for the financial n=26 workload. Commitment cost is linear in provisioned bytes and independent of the trial count, so it is amortized by larger test sets and is the term that dominates at small ones. Certification transmitted 342.43 KiB for the confidence n=8 workload, and 342.43 KiB for the confidence n=24 workload, and 342.45 KiB for the financial n=26 workload of certifier-workload protocol traffic on the in-process socket pair. Almost all of that is one message: the sealed test environment, padded to a fixed length so its size reveals nothing about the secret test, accounts for 341.42 KiB, and 341.42 KiB, and 341.42 KiB, leaving 1.01 KiB of commit and certificate traffic for the confidence n=8 workload, and 1.01 KiB of commit and certificate traffic for the confidence n=24 workload, and 1.03 KiB of commit and certificate traffic for the financial n=26 workload. The environment component is invariant to the workload and to the trial count by construction, so it is a property of the protocol rather than a per-evaluation cost.

## Per workload

### confidence n=8

- environment `treecut-generated`, 8 trials, seed 7
- 5 repeats x 2 orderings = 10 processes, each running the evaluation twice
- model `google/gemma-4-E2B-it` at cuda/auto
- released m(FN)/m_fp = 0.5000 / 0.0000 — identical in both arms of every process, which is the comparison's control, not a probe-quality result
- arms consistent across all processes: True
- first-arm host peak RSS: direct 10.615 GiB, certified 10.615 GiB

### confidence n=24

- environment `treecut-generated`, 24 trials, seed 7
- 5 repeats x 2 orderings = 10 processes, each running the evaluation twice
- model `google/gemma-4-E2B-it` at cuda/auto
- released m(FN)/m_fp = 0.5000 / 0.0000 — identical in both arms of every process, which is the comparison's control, not a probe-quality result
- arms consistent across all processes: True
- first-arm host peak RSS: direct 10.611 GiB, certified 10.615 GiB

### financial n=26

- environment `apollo-insider-trading`, 26 trials, seed 7
- 5 repeats x 2 orderings = 10 processes, each running the evaluation twice
- model `google/gemma-2-2b-it` at cuda/bfloat16, attn `eager`
- released m(FN)/m_fp = 0.0000 / 0.8462 — identical in both arms of every process, which is the comparison's control, not a probe-quality result
- arms consistent across all processes: True
- first-arm host peak RSS: direct 5.682 GiB, certified 5.682 GiB

## Glossary

Terms as used above, stated precisely because the measurement design depends
on the distinctions.

**Quantities**

| term | field | definition |
|---|---|---|
| commitment input | `commitment_input_bytes` | Total bytes fed to the certificate commitment — every provisioned artifact file that gets hashed. Fixed by the model and probe, independent of the test set. |
| commitment time | `commitment_seconds` | Wall time to compute the certificate commitment over those bytes. Hashed twice per arm (once, then again to detect a file changing while its loader consumed it). |
| model run duration | `model_run_seconds` | Wall time executing the trials: the agent generating and the tap scoring. Scales with the trial count. |
| protocol residual | wall − commitment − model run | The protocol's own work: sealing, socket transfer, signing, verification. Measured inside one arm rather than by differencing arms. |
| transmitted total | `transmitted_bytes_total` | All bytes crossing the certifier/workload socket pair. **Not** a compressed certificate — see the split below. |
| ↳ environment | `transmitted_environment_bytes` | The secret test environment sent **to** the workload, sealed and padded to a fixed length so its size leaks nothing. ~99.7% of the total, and invariant to workload and trial count. |
| ↳ commit | `transmitted_commit_bytes` | The commit message from workload to certifier, carrying the commitment digest `h_c` (32 bytes) plus signature. |
| ↳ certificate | `transmitted_certificate_bytes` | The certificate itself, workload to certifier: `(m, m_fp)`, `h_c`, `h_s`, `t_E`, signature. |

**Measurement structure**

| term | definition |
|---|---|
| arm | One execution of the evaluation, either `direct` (no protocol) or `certified` (the same evaluation driven through the trusted-workload/certifier protocol). |
| **paired run** | One OS process that loads the model **once**, then runs both arms back to back. The shared model load is what makes the two arms comparable; each paired run yields one paired observation. Previously called "one process". |
| arm order | Which arm runs first inside a paired run — `direct_first` or `certified_first`. Counterbalanced so once-per-process warm-up (CUDA context, allocator, autotuning) does not land on one arm. |
| repeat | Index of a paired run within one arm order. |
| **measurement set** | All paired runs sharing one workload configuration (model, dtype, trial count, seed, token budget): `2 × repeats` paired runs. Medians are taken over this set. Previously called "campaign". |
| session | Everything run between one instance start and stop. Relevant because a stop/start can reschedule onto different physical hardware, which would confound pooled timing data. |

## Scope

- Not a TEE measurement. The certifier and the trusted workload are two protocol endpoints in one process on a general-purpose VM; attestation and confidential provisioning are not part of this.
- `wire` is the exact byte count on that in-process socket pair, not a network measurement. It is expected to be near-constant across workloads: the environment crosses the boundary padded to a fixed length so its size leaks nothing, and that padding dominates the total.
- Host RSS excludes device memory; only the first arm of each process has a usable high-water mark.
- `(m, m_fp)` appears as the control that both arms agree, not as a result. For the financial workload the threshold is uncalibrated and the concealment label is the repo's heuristic judge.
