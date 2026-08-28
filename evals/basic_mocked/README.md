# `evals/basic_mocked/` — the protocol on mock models, no GPU

The whole Certified Taps pipeline end to end with no model and no GPU, so you can see
the protocol's behavior in seconds. The program and taps are scripted stand-ins (not
evidence of anything), running on real TreeCut items. Real-model evidence comes from
[`evals/confidence`](../../evals/confidence) and [`evals/financial`](../../evals/financial).

## What you'll see

- **Honest run.** The certifier prints `CERTIFIED  m(FN)=0.000  m_fp=0.000`, a VALID
  signature, and `separation (FN) +0.625`. The tap reads a hesitation the mock plants
  in the thought channel on unanswerable items and misses none of them; the black-box
  comparator sees only the answer and misses most, so it lands 0.625 worse. That gap is
  the introspection advantage the paper is about, in miniature. The mock's numbers are
  clean by construction, so read the +0.625 as "the pipeline runs and the gap is
  measurable," not as an effect size.
- **Environment boundary.** The fixture crosses the wire as fixed-length inert data;
  the workload authenticates it before parsing and executes only its registered static
  driver. No fixture object or executable code is deserialized from C.
- **Tamper run (`--tamper-environment`).** The relay replaces C's environment message
  with a fresh encryption of the same E. The enclave can execute it, but C rejects the
  resulting certificate because `t_E` does not name C's ciphertext. That is the
  protocol's exact-message integrity property made observable, not a crash.
- **Gaming demo.** An honest tap and one that memorized the public benchmark both score
  0.0 on the public items. On the freshly generated secret test the honest tap stays at
  0.0 while the gamed one jumps to 0.5, and the certificate reports that true
  secret-test loss. This checks that the harness exposes exact-item memorization on a
  disjoint fixture; it does not prove resistance to adaptive gaming.
- **Comparison record.** Runs the same seeded tests directly and through the certified
  in-process socket path, checks that their aggregate measurements match, and writes
  local runtime/resource/traffic metrics as JSON.

## References

- The mock gaming demo is a deterministic protocol sanity check, not efficacy evidence.
- Test items come from the shared TreeCut generator in
  [`util/treecut`](../../util/treecut) (arXiv:2502.13442). A `z = 1` item had its
  solution path cut, so a confident number is a hallucination.

## Run it

The honest run, the tamper demo, the gaming demo, and the separate-process socket
rehearsal are all in
[RUNBOOK Phase 0](../../RUNBOOK.md#phase-0--get-the-mock-harness-running-5-min-no-models)
(the rehearsal is under Phase 3).
