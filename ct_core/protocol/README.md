# `ct_core/protocol/` — cryptography across the trust boundary

The sealing, digests, and attestation used by `ct_core/certification.py`. `run_session`
connects the workload and certifier with a socket pair; the separate-process rehearsal
can relay the same three messages through an untrusted host.

Attestation uses separate interfaces. Local setup pins the verification key and
expected workload measurement before the workload starts. `StubEnclaveSigner` can
then only load the signing key; it cannot create or change that public trust anchor.
C and R load only the pinned public record through `StubAttestationVerifier`. The
corresponding GCP signer and verifier remain the hardware-deployment seams.

The provider provisions separate inert agent and tap packages for drivers registered
in the measured workload. The workload hashes their canonical configuration and actual
artifact files/bytes itself; each driver creates a fresh trial-session object.
Provider-supplied Python never executes in the trusted process.

The wire carries one fixed-length encrypted environment, never a Python object. The
workload checks its public `id_E` before JSON parsing and accepts only a driver whose
exact source is both packaged by C and registered in the measured workload. Supporting
arbitrary C-supplied environment code requires a TEE-side sandbox or interpreter.

Where its behavior surfaces: the `CERTIFIED` line with a VALID signature, and the
certifier abort when a relay substitutes the environment ciphertext, both in
[`evals/basic_mocked`](../../evals/basic_mocked). The exact message schema and the
execution path are in `ct_core/certification.py`.

## References

- libsodium sealed boxes via PyNaCl for the confidential transport; Ed25519 for
  attestation.
- The paper's `Prot_CP` and its realization proof, which this layer implements.

## Run it

Nothing to run directly. Local checks are in
[RUNBOOK Phase 0](../../RUNBOOK.md#phase-0--get-the-mock-harness-running-5-min-no-models);
remaining Confidential Space work is in
[Phase 3](../../RUNBOOK.md#phase-3--google-confidential-space-deployment-later).
