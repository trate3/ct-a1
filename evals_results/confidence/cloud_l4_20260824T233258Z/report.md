# Cost of certification — Compute Engine L4

- instance: `g2-standard-8` in `us-west1-a` (None), NVIDIA L4 22 GiB, driver 580.173.02, CUDA 13.0
- torch 2.13.0, transformers 5.15.1, python 3.11.16

| workload | trials | repeats | wall direct | wall certified | Δwall | CPU direct | CPU certified | ΔCPU | GPU peak Δ | wire |
|---|---|---|---|---|---|---|---|---|---|---|
| confidence | 24 | 5 | 571.84 s | 562.73 s | -9.11 s (-1.6%) | 571.84 s | 562.75 s | -9.09 s (-1.6%) | +4,160 KiB | 342.43 KiB (1.01 control + 341.42 env) |

## Where the time goes

Both arms commit the provider's artifacts, so that cost cancels out of the direct-vs-certified delta above and is invisible in it — while remaining a real, and here dominant, cost of certifying at all.

| workload | arm | commit artifacts | evaluate trials | residual | artifacts | hash throughput |
|---|---|---|---|---|---|---|
| confidence | direct | 116.42 s | 455.40 s | **0.000 s** ± 0.000 | 9.57 GiB | 177 MB/s (2x) |
|  | certified | 116.13 s | 446.46 s | **0.220 s** ± 0.017 |  |  |

The direct arm's residual is the harness's zero point: it commits and evaluates and does nothing else. The certified arm's residual is the protocol itself — sealing, the socket transfer, signing and verification — measured inside one arm instead of by subtracting two multi-minute arms to find a sub-second quantity.

## Reported paragraph

At the tested workload sizes on a single NVIDIA L4, certification added little overhead to the confidence workload. For confidence evaluation, median wall time decreased from 571.84 s to 562.73 s (-1.6%), while CPU time decreased from 571.84 s to 562.75 s (-1.6%). Process-lifetime peak host RSS did not measurably change for confidence evaluation; under GPU execution the model weights reside in device memory and are not counted in host RSS. Median peak CUDA memory allocated changed by +4,160 KiB for confidence evaluation. Differencing the two arms cannot resolve the protocol's cost at this workload size: the run-to-run spread is larger than the quantity, and in these runs it is larger than it with the opposite sign. Measured instead inside the certified arm, as the wall time left after committing the artifacts and running the trials, the protocol accounts for 0.220 s (0.039% of the confidence workload), against a direct-arm residual of 0.000 s. Those deltas exclude the dominant cost, because both arms pay it: committing the provider's artifacts means hashing every provisioned file, which took a median 116.42 s against 455.40 s of actual trials for the confidence workload. Commitment cost is linear in provisioned bytes and independent of the trial count, so it is amortized by larger test sets and is the term that dominates at small ones. Certification transmitted 342.43 KiB for the confidence workload of certifier-workload protocol traffic on the in-process socket pair. Almost all of that is one message: the sealed test environment, padded to a fixed length so its size reveals nothing about the secret test, accounts for 341.42 KiB, leaving 1.01 KiB of commit and certificate traffic for the confidence workload. The environment component is invariant to the workload and to the trial count by construction, so it is a property of the protocol rather than a per-evaluation cost.

## Per workload

### confidence

- environment `treecut-generated`, 24 trials, seed 7
- 5 repeats x 2 orderings = 10 processes, each running the evaluation twice
- model `google/gemma-4-E2B-it` at cuda/auto
- released m(FN)/m_fp = 0.5000 / 0.0000 — identical in both arms of every process, which is the comparison's control, not a probe-quality result
- arms consistent across all processes: True
- first-arm host peak RSS: direct 10.611 GiB, certified 10.615 GiB

## Scope

- Not a TEE measurement. The certifier and the trusted workload are two protocol endpoints in one process on a general-purpose VM; attestation and confidential provisioning are not part of this.
- `wire` is the exact byte count on that in-process socket pair, not a network measurement. It is expected to be near-constant across workloads: the environment crosses the boundary padded to a fixed length so its size leaks nothing, and that padding dominates the total.
- Host RSS excludes device memory; only the first arm of each process has a usable high-water mark.
- `(m, m_fp)` appears as the control that both arms agree, not as a result. For the financial workload the threshold is uncalibrated and the concealment label is the repo's heuristic judge.
