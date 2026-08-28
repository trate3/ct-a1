"""Verifier B: holds the secret test environment, checks the certificate.

(The trusted auditor of the paper's running example is this verifier; "verifier"
is the one name used in code.)

The verifier drives the test environment, seals every query and the labels to the
enclave's channel key, and for every reply: recomputes the ciphertext-transcript
digest from the ciphertexts IT sent, verifies the attestation signature, then opens
the sealed response with its own response key. Any mismatch -- a prover that
substitutes or re-seals a query, forges a response, or tampers with the certificate
-- aborts. On success it returns the certificate and its own record of the tests.

For a chained session it also receives the referenced certificates, verifies each
chain and its exact digest at commit time, and only then asks the environment for its
first secret query.

Raises `VerifierAbort` on any protocol violation.
"""

from __future__ import annotations

import socket

from ..config import PROTOCOL_VERSION
from ..interfaces import TestEnvironment
from ..protocol import sealing, wire
from ..protocol.attestor import Attestor, StubAttestor
from ..protocol.digests import certificate_digest, signing_payload, tests_digest, transcript_digest
from .check_certificate import certificate_valid


class VerifierAbort(Exception):
    """Raised when the verifier rejects the session."""


def run_verifier(
    sock: socket.socket,
    env: TestEnvironment,
    attestor: Attestor | None = None,
    progress=None,
    dependency_certificates: tuple[dict, ...] = (),
) -> dict:
    """Run one certification session as the verifier; return the certificate dict.

    `progress`, if given, is called `progress(round_i, n)` at the start of each round,
    so a caller running a slow real model can show that the session is advancing.
    """
    attestor = attestor or StubAttestor()
    reader = wire.LineReader(sock)

    # --- commit ---
    commit, _ = reader.recv()
    if not commit or commit["type"] != wire.COMMIT:
        raise VerifierAbort("no commit received")
    dependencies = commit.get("dependencies", [])
    if (not isinstance(dependencies, list)
            or any(not isinstance(d, str) for d in dependencies)
            or len(set(dependencies)) != len(dependencies)):
        raise VerifierAbort("malformed certificate dependencies")
    if not attestor.verify(
            signing_payload("commit", commit["version"], commit["m_ref"], commit["N"],
                            commit["h_c"], commit["pk_ch"], commit["n_rounds"],
                            dependencies), commit["sig"]):
        raise VerifierAbort("commit attestation failed")
    if commit["version"] != PROTOCOL_VERSION:
        raise VerifierAbort("protocol version mismatch")
    if commit["n_rounds"] != env.n():
        raise VerifierAbort("round-count mismatch")
    if commit["m_ref"] != attestor.measurement():
        raise VerifierAbort("measurement mismatch (unexpected enclave code)")
    try:
        expected_dependencies = [certificate_digest(cert)
                                 for cert in dependency_certificates]
    except (KeyError, TypeError, ValueError) as exc:
        raise VerifierAbort("malformed upstream certificate") from exc
    if dependencies != expected_dependencies:
        raise VerifierAbort("certificate dependency mismatch")
    if any(not certificate_valid(cert, dependency_certificates, attestor)
           for cert in dependency_certificates):
        raise VerifierAbort("upstream certificate verification failed")
    m_ref, nonce, h_c = commit["m_ref"], commit["N"], commit["h_c"]
    pk_ch = bytes.fromhex(commit["pk_ch"])

    # Our ephemeral response key: the enclave seals responses/certout to pk_B.
    sk_B, pk_B = sealing.new_keypair()

    # --- interactive testing ---
    sent_ciphertexts: list[bytes] = []   # raw bytes of every query/label ciphertext, for t_i
    transcript: list[dict] = []
    for i in range(1, env.n() + 1):
        if progress:
            progress(i, env.n())
        x = env.next_query(i)
        payload = {"i": i, "x": x}
        if i == 1:
            payload["pk_B"] = pk_B.hex()
        c = sealing.seal_json(pk_ch, payload)
        sent_ciphertexts.append(sealing.ciphertext_bytes(c))
        wire.send(sock, {"type": wire.QUERY, "i": i, "c": c})

        resp, _ = reader.recv()
        if not resp or resp.get("type") != wire.RESPONSE or resp.get("i") != i:
            raise VerifierAbort(f"round {i}: missing or malformed response")
        t_i = transcript_digest(sent_ciphertexts)
        if resp["t_i"] != t_i:
            raise VerifierAbort(f"round {i}: transcript digest mismatch (query substitution)")
        if not attestor.verify(
                signing_payload("response", PROTOCOL_VERSION, m_ref, nonce, i,
                                resp["d"], t_i), resp["sig"]):
            raise VerifierAbort(f"round {i}: response attestation failed")
        inner = sealing.unseal_json(sk_B, resp["d"])
        if inner.get("i") != i or inner.get("t_i") != t_i:
            raise VerifierAbort(f"round {i}: sealed response does not match")
        y = inner["y"]
        env.observe(i, y)
        transcript.append({"i": i, "x": x, "y": y})

    # --- certify ---
    labels = env.labels()
    c_z = sealing.seal_json(pk_ch, {"labels": labels})
    sent_ciphertexts.append(sealing.ciphertext_bytes(c_z))
    wire.send(sock, {"type": wire.CERTIFY, "c": c_z})

    certout, _ = reader.recv()
    if not certout or certout.get("type") != wire.CERTOUT:
        raise VerifierAbort("no certify output")
    inner = sealing.unseal_json(sk_B, certout["d"])
    tests = [(t["x"], z) for t, z in zip(transcript, labels)]
    t_cert = transcript_digest(sent_ciphertexts)
    if (inner.get("version") != PROTOCOL_VERSION or inner.get("n_rounds") != env.n()
            or inner.get("t_cert") != t_cert
            or inner.get("h_T") != tests_digest(tests)):
        raise VerifierAbort("certify digests mismatch")
    if not attestor.verify(
            signing_payload("certify", PROTOCOL_VERSION, env.n(), m_ref, nonce, h_c,
                            inner["h_T"], inner["t_cert"], inner["m"], inner["m_fp"],
                            dependencies), certout["sig"]):
        raise VerifierAbort("certificate signature failed")

    return {
        "version": PROTOCOL_VERSION, "n_rounds": env.n(),
        "m_ref": m_ref, "N": nonce, "h_c": h_c, "h_T": inner["h_T"],
        "t_cert": inner["t_cert"], "m": inner["m"], "m_fp": inner["m_fp"],
        "dependencies": dependencies, "sig": certout["sig"],
        "_reveal": {"tests": tests, "transcript": transcript},  # verifier-only record
    }
