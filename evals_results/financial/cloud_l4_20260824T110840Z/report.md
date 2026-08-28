# Cost of certification — Compute Engine L4

- instance: `g2-standard-8` in `us-west1-a` (None), NVIDIA L4 22 GiB, driver 580.173.02, CUDA 13.0
- torch 2.13.0, transformers 5.15.1, python 3.11.16

| workload | trials | repeats | wall direct | wall certified | Δwall | CPU direct | CPU certified | ΔCPU | GPU peak Δ | wire |
|---|---|---|---|---|---|---|---|---|---|---|
| financial | 26 | 5 | 382.72 s | 373.95 s | -8.77 s (-2.3%) | 382.88 s | 374.11 s | -8.78 s (-2.3%) | +4,722 KiB | 342.45 KiB (1.03 control + 341.42 env) |

## Where the time goes

Both arms commit the provider's artifacts, so that cost cancels out of the direct-vs-certified delta above and is invisible in it — while remaining a real, and here dominant, cost of certifying at all.

| workload | arm | commit artifacts | evaluate trials | residual | artifacts | hash throughput |
|---|---|---|---|---|---|---|
| financial | direct | 29.72 s | 352.92 s | **0.000 s** ± 0.000 | 4.89 GiB | 353 MB/s (2x) |
|  | certified | 29.71 s | 344.22 s | **0.016 s** ± 0.000 |  |  |

The direct arm's residual is the harness's zero point: it commits and evaluates and does nothing else. The certified arm's residual is the protocol itself — sealing, the socket transfer, signing and verification — measured inside one arm instead of by subtracting two multi-minute arms to find a sub-second quantity.

## Reported paragraph

At the tested workload sizes on a single NVIDIA L4, certification added moderate overhead to the financial workload. For financial evaluation, median wall time decreased from 382.72 s to 373.95 s (-2.3%), while CPU time decreased from 382.88 s to 374.11 s (-2.3%). Process-lifetime peak host RSS did not measurably change for financial evaluation; under GPU execution the model weights reside in device memory and are not counted in host RSS. Median peak CUDA memory allocated changed by +4,722 KiB for financial evaluation. Differencing the two arms cannot resolve the protocol's cost at this workload size: the run-to-run spread is larger than the quantity, and in these runs it is larger than it with the opposite sign. Measured instead inside the certified arm, as the wall time left after committing the artifacts and running the trials, the protocol accounts for 0.015 s (0.004% of the financial workload), against a direct-arm residual of 0.000 s. Those deltas exclude the dominant cost, because both arms pay it: committing the provider's artifacts means hashing every provisioned file, which took a median 29.72 s against 352.92 s of actual trials for the financial workload. Commitment cost is linear in provisioned bytes and independent of the trial count, so it is amortized by larger test sets and is the term that dominates at small ones. Certification transmitted 342.45 KiB for the financial workload of certifier-workload protocol traffic on the in-process socket pair. Almost all of that is one message: the sealed test environment, padded to a fixed length so its size reveals nothing about the secret test, accounts for 341.42 KiB, leaving 1.03 KiB of commit and certificate traffic for the financial workload. The environment component is invariant to the workload and to the trial count by construction, so it is a property of the protocol rather than a per-evaluation cost.

## Per workload

### financial

- environment `apollo-insider-trading`, 26 trials, seed 7
- 5 repeats x 2 orderings = 10 processes, each running the evaluation twice
- model `google/gemma-2-2b-it` at cuda/bfloat16, attn `eager`
- released m(FN)/m_fp = 0.0000 / 0.8462 — identical in both arms of every process, which is the comparison's control, not a probe-quality result
- arms consistent across all processes: True
- first-arm host peak RSS: direct 5.682 GiB, certified 5.682 GiB

## Scope

- Not a TEE measurement. The certifier and the trusted workload are two protocol endpoints in one process on a general-purpose VM; attestation and confidential provisioning are not part of this.
- `wire` is the exact byte count on that in-process socket pair, not a network measurement. It is expected to be near-constant across workloads: the environment crosses the boundary padded to a fixed length so its size leaks nothing, and that padding dominates the total.
- Host RSS excludes device memory; only the first arm of each process has a usable high-water mark.
- `(m, m_fp)` appears as the control that both arms agree, not as a result. For the financial workload the threshold is uncalibrated and the concealment label is the repo's heuristic judge.
