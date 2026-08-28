"""Canonical hashes and signature encoding."""

from __future__ import annotations

import hashlib
import json


def sha256_hex(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(len(p).to_bytes(8, "big"))
        h.update(p)
    return h.hexdigest()


def artifact_commitment_digest(certified_object: bytes, r: bytes) -> str:
    """h_c over the trusted workload's encoding of the actual provisioned artifacts."""
    return sha256_hex(certified_object, r)


def tests_digest(tests: list[tuple[str, int]]) -> str:
    """Offline-only identity for a labeled evaluation set."""
    canonical = json.dumps([list(t) for t in tests], sort_keys=True, separators=(",", ":"))
    return sha256_hex(canonical.encode())


def signing_payload(*fields) -> bytes:
    """Canonical byte encoding of a tuple of signed fields."""
    return json.dumps(list(fields), sort_keys=True, separators=(",", ":")).encode()


# ---- ct/0.1 digests, for the grafted harness/roles path (RQ2 conflict case) ----
# ct/0.2 commits the provisioned artifact BYTES (`artifact_commitment_digest` over the
# workload's encoding of the files it actually loaded). ct/0.1 commits the digests the
# live objects report about themselves. The two are not interchangeable: the first is
# what makes commitment cost scale with checkpoint size, which is the dominant term in
# the RQ1 measurements, and the second trusts the objects to describe themselves.


def commitment_digest(model_digest: str, control_digest: str, tools_digest: str,
                      tap_digest: str, r: bytes) -> str:
    """h_c over the digests the committed agent and tap report (ct/0.1)."""
    return sha256_hex(model_digest.encode(), control_digest.encode(),
                      tools_digest.encode(), tap_digest.encode(), r)


def transcript_digest(ciphertexts: list[bytes]) -> str:
    """Running digest over the exact ciphertexts a role received."""
    return sha256_hex(*ciphertexts)


def certificate_digest(cert: dict) -> str:
    """Stable identity for a certificate, used to reference one from another."""
    canonical = json.dumps(cert, sort_keys=True, separators=(",", ":")).encode()
    return sha256_hex(canonical)
