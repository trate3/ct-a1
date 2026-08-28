"""Enclave evaluation service -- the trusted workload (Prot_CP, confidential).

`serve_session` runs one certification session over a confidential channel:

    commit  ->  n x (query -> response)  ->  certify  ->  certout

Confidentiality (the relaying prover sees only ciphertext it cannot read):
  - at commit the enclave generates an ephemeral channel keypair (pk_ch, sk_ch) and
    puts pk_ch in the attested commit; the verifier seals its queries and labels to
    pk_ch;
  - the verifier's first query carries its response key pk_B; the enclave seals every
    response and the certify output to pk_B.
Integrity against a prover that re-seals its own query (it can, pk_ch is public):
  - each response carries t_i = H(ciphertexts received so far). The verifier recomputes
    t_i over the ciphertexts IT sent; any substitution makes t_i mismatch and the
    verifier aborts -- so integrity is ciphertext-level, not from the sealing.

The tap's per-round prediction and the program's thought channel never leave this
process; r never leaves; only the aggregate (m, m_fp) is released, signed.

This is a library (no CLI). A concrete enclave that wires a specific program + tap
is provided by each eval (e.g. evals/basic_mocked/enclave_cli.py).
"""

from __future__ import annotations

import secrets
import socket

from ..agent import Agent
from ..config import NONCE_BYTES, PROTOCOL_VERSION, RANDOMIZER_BYTES
from ..interfaces import Monitor, MonitorInput
from ..protocol import sealing, wire
from ..protocol.attestor import Attestor, StubAttestor
from ..protocol.digests import (
    certificate_digest,
    commitment_digest,
    signing_payload,
    tests_digest,
    transcript_digest,
)


def serve_session(
    conn: socket.socket,
    agent: Agent,
    tap: Monitor,
    n_rounds: int,
    attestor: Attestor | None = None,
    commitment_randomizer: bytes | None = None,
    dependency_certificates: tuple[dict, ...] = (),
) -> None:
    """Run one certification session over an already-connected socket."""
    if isinstance(n_rounds, bool) or not isinstance(n_rounds, int) or n_rounds <= 0:
        raise ValueError("n_rounds must be a positive integer")
    attestor = attestor or StubAttestor()
    m_ref = attestor.measurement()
    nonce = secrets.token_hex(NONCE_BYTES)              # session nonce N
    if commitment_randomizer is not None and (
            not isinstance(commitment_randomizer, bytes)
            or len(commitment_randomizer) != RANDOMIZER_BYTES):
        raise ValueError(f"commitment randomizer must be {RANDOMIZER_BYTES} bytes")
    # Normally generated here.  For a supply-chain component certificate, the provider
    # chooses r and later carries it in the private opening.  It is never sent to the
    # verifier or included in the public certificate.
    r = (commitment_randomizer if commitment_randomizer is not None
         else secrets.token_bytes(RANDOMIZER_BYTES))
    dependencies = [certificate_digest(cert) for cert in dependency_certificates]
    # Commit the AGENT: its three components (model, control-logic, tools) are bound
    # explicitly, so the certificate binds the whole agent, not just the model.
    h_c = commitment_digest(agent.model_digest(), agent.control_digest(),
                            agent.tools_digest(), tap.digest(), r)

    # Ephemeral channel keypair: the verifier seals queries/labels to pk_ch.
    sk_ch, pk_ch = sealing.new_keypair()
    pk_ch_hex = pk_ch.hex()

    sig_c = attestor.sign(
        signing_payload("commit", PROTOCOL_VERSION, m_ref, nonce, h_c, pk_ch_hex,
                        n_rounds, dependencies))
    wire.send(conn, {"type": wire.COMMIT, "version": PROTOCOL_VERSION,
                     "m_ref": m_ref, "N": nonce, "h_c": h_c,
                     "pk_ch": pk_ch_hex, "n_rounds": n_rounds,
                     "dependencies": dependencies, "sig": sig_c})

    reader = wire.LineReader(conn)
    recv_ciphertexts: list[bytes] = []   # raw bytes of every query/label ciphertext, for t_i
    xs: list[str] = []
    zhats: list[int] = []
    pk_B: bytes | None = None            # verifier's response key, learned in round 1
    ctr = 0

    # Only capture activations when the tap is a white-box probe, and from the
    # program's own execution (same-execution binding).
    capture = tap.probe_layers if getattr(tap, "needs_activations", False) else None

    while True:
        msg, _ = reader.recv()
        if msg is None:
            return  # peer closed -> silent abort

        if msg["type"] == wire.QUERY:
            c = msg.get("c")
            if not isinstance(c, str):
                return
            try:
                payload = sealing.unseal_json(sk_ch, c)   # authenticated; raises on tamper
            except Exception:
                return  # undecryptable ciphertext -> silent abort
            recv_ciphertexts.append(sealing.ciphertext_bytes(c))
            i = payload.get("i")
            x = payload.get("x")
            # Typed checks mirroring the ideal functionality: round order, budget, admissibility.
            if i != ctr + 1 or ctr >= n_rounds or not isinstance(x, str):
                return  # silent abort
            if ctr == 0:
                pkb = payload.get("pk_B")
                if not isinstance(pkb, str):
                    return
                pk_B = bytes.fromhex(pkb)
            ctr = i
            gen = agent.generate(x, capture_layers=capture)     # one execution; thought stays here
            pred = tap.predict(MonitorInput(x=x, answer=gen.answer, thought=gen.thought,
                                            activations=gen.activations))
            xs.append(x)
            zhats.append(pred.z_hat)
            t_i = transcript_digest(recv_ciphertexts)           # over ciphertexts received
            d = sealing.seal_json(pk_B, {"i": ctr, "y": gen.answer, "t_i": t_i})
            sig_i = attestor.sign(
                signing_payload("response", PROTOCOL_VERSION, m_ref, nonce, ctr, d, t_i))
            wire.send(conn, {"type": wire.RESPONSE, "i": ctr, "d": d, "t_i": t_i, "sig": sig_i})

        elif msg["type"] == wire.CERTIFY:
            c_z = msg.get("c")
            if not isinstance(c_z, str) or ctr != n_rounds:  # exact-n: certify only after all rounds
                return  # silent abort
            try:
                labels = sealing.unseal_json(sk_ch, c_z).get("labels")
            except Exception:
                return
            if (not isinstance(labels, list) or len(labels) != ctr
                    or any(z not in (0, 1) for z in labels)):
                return
            recv_ciphertexts.append(sealing.ciphertext_bytes(c_z))
            tests = list(zip(xs, labels))
            h_T = tests_digest(tests)
            t_cert = transcript_digest(recv_ciphertexts)        # over all query + label ciphertexts
            n = ctr
            m = sum(1 for zh, z in zip(zhats, labels) if z == 1 and zh == 0) / n     # FN rate
            m_fp = sum(1 for zh, z in zip(zhats, labels) if z == 0 and zh == 1) / n  # FP rate
            sig_m = attestor.sign(
                signing_payload("certify", PROTOCOL_VERSION, n_rounds, m_ref, nonce,
                                h_c, h_T, t_cert, m, m_fp, dependencies))
            d_cert = sealing.seal_json(
                pk_B, {"version": PROTOCOL_VERSION, "n_rounds": n_rounds,
                       "h_T": h_T, "t_cert": t_cert, "m": m, "m_fp": m_fp})
            wire.send(conn, {"type": wire.CERTOUT, "d": d_cert, "sig": sig_m})
            return

        else:
            return  # unknown message type -> silent abort
