"""Confidential transport: anonymous encryption to a public key (sealed boxes).

The certifier encrypts E once to the enclave's ephemeral channel key. An optional
untrusted relay sees only that ciphertext and the public certificate.

We use libsodium sealed boxes via PyNaCl (`SealedBox`): X25519 key agreement with
a fresh ephemeral sender key + XSalsa20-Poly1305 AEAD -- anonymous (the recipient
public key is all you need to encrypt) and IND-CCA2, which is the property the
protocol's realization proof relies on. Anyone can seal to the public key; `t_E`
therefore binds the certificate to the exact ciphertext the certifier sent.

Ciphertext travels the wire as base64 text inside JSON messages; digests are
computed over the raw ciphertext bytes on both sides.
"""

from __future__ import annotations

import base64
import json
from nacl.public import PrivateKey, PublicKey, SealedBox


def new_keypair() -> tuple[PrivateKey, bytes]:
    """Fresh ephemeral keypair; returns (secret key object, public-key bytes for the wire)."""
    sk = PrivateKey.generate()
    return sk, bytes(sk.public_key)


def seal_bytes(pk_bytes: bytes, plaintext: bytes) -> str:
    """Seal fixed-length bytes; return base64 ciphertext for the JSON wire."""
    ct = SealedBox(PublicKey(pk_bytes)).encrypt(plaintext)
    return base64.b64encode(ct).decode()


def unseal_bytes(sk: PrivateKey, b64_ciphertext: str) -> bytes:
    """Open a sealed ciphertext with the matching private key (raises on tamper)."""
    return SealedBox(sk).decrypt(base64.b64decode(b64_ciphertext, validate=True))


def ciphertext_bytes(b64_ciphertext: str) -> bytes:
    """The raw ciphertext bytes bound by t_E."""
    return base64.b64decode(b64_ciphertext, validate=True)


# ---- ct/0.1 JSON sealing, for the grafted harness/roles path (RQ2 conflict case) ----
# ct/0.2 seals one fixed-length byte blob: the whole environment, padded so its size
# leaks nothing. ct/0.1 seals a JSON payload per round instead, which is why that path
# has per-round ciphertexts to hash into a transcript digest.


def _canonical(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def seal_json(pk_bytes: bytes, obj: dict) -> str:
    """Seal a JSON payload to a public key; returns base64 ciphertext for the wire."""
    ct = SealedBox(PublicKey(pk_bytes)).encrypt(_canonical(obj))
    return base64.b64encode(ct).decode()


def unseal_json(sk: PrivateKey, b64_ciphertext: str) -> dict:
    """Open a sealed base64 ciphertext with the matching secret key (raises on tamper)."""
    return json.loads(SealedBox(sk).decrypt(base64.b64decode(b64_ciphertext)))
