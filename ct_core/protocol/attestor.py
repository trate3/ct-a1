"""Attestation signing and public verification across separate trust domains.

Setup publishes the verification key and expected workload measurement before any
session. The trusted workload owns an ``EnclaveSigner``; the certifier and relying
party own only an ``AttestationVerifier`` initialized from that pinned publication.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import secrets

from nacl.signing import SigningKey, VerifyKey

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
_KEY_DIR = _REPO_ROOT / ".keys"
_SEED_PATH = _KEY_DIR / "manufacturer.key"
_PUBLICATION_PATH = _KEY_DIR / "attestation.json"
_MEASURE_GLOB = "*.py"


def _workload_measurement() -> str:
    """Local MRENCLAVE analog: a deterministic hash of the trusted package."""
    digest = hashlib.sha256()
    for path in sorted(_PACKAGE_ROOT.rglob(_MEASURE_GLOB)):
        relative = path.relative_to(_PACKAGE_ROOT).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _create_signing_key() -> SigningKey:
    """Manufacturer-side setup for the local hardware-key stand-in."""
    _KEY_DIR.mkdir(exist_ok=True)
    if not _SEED_PATH.exists():
        temporary = _KEY_DIR / f".seed.tmp.{os.getpid()}.{secrets.token_hex(4)}"
        temporary.write_bytes(bytes(SigningKey.generate()))
        try:
            os.link(temporary, _SEED_PATH)
        except FileExistsError:
            pass
        finally:
            temporary.unlink(missing_ok=True)
    return SigningKey(_SEED_PATH.read_bytes())


def _publish(verify_key: bytes, measurement: str) -> None:
    """Atomically publish the two values available to C and R in the paper."""
    publication = json.dumps({
        "verify_key": verify_key.hex(),
        "expected_measurement": measurement,
    }, sort_keys=True).encode()
    temporary = _KEY_DIR / f".attestation.tmp.{os.getpid()}.{secrets.token_hex(4)}"
    temporary.write_bytes(publication)
    os.replace(temporary, _PUBLICATION_PATH)


def initialize_local_attestation() -> "StubAttestationVerifier":
    """Setup authority: pin public verification material before any workload runs."""
    signing_key = _create_signing_key()
    verifier = StubAttestationVerifier(
        bytes(signing_key.verify_key), _workload_measurement())
    _publish(bytes(signing_key.verify_key), verifier.measurement())
    return verifier


class EnclaveSigner:
    """Trusted-workload interface; implementations hold the attestation secret."""

    def measurement(self) -> str:
        raise NotImplementedError

    def sign(self, payload: bytes) -> str:
        raise NotImplementedError


class AttestationVerifier:
    """Public certifier/relying-party interface; it deliberately cannot sign."""

    def measurement(self) -> str:
        raise NotImplementedError

    def verify(self, payload: bytes, signature: str) -> bool:
        raise NotImplementedError


class StubEnclaveSigner(EnclaveSigner):
    """Local enclave stand-in; it cannot create or update the public trust anchor."""

    def __init__(self, signing_key: SigningKey | None = None) -> None:
        self._signing_key = signing_key or SigningKey(_SEED_PATH.read_bytes())
        self._measurement = _workload_measurement()

    def measurement(self) -> str:
        return self._measurement

    def sign(self, payload: bytes) -> str:
        return self._signing_key.sign(payload).signature.hex()


class StubAttestationVerifier(AttestationVerifier):
    """Local C/R verifier containing only public verification material."""

    def __init__(self, verify_key: bytes, expected_measurement: str) -> None:
        self._verify_key = VerifyKey(verify_key)
        self._measurement = expected_measurement

    @classmethod
    def from_publication(cls) -> "StubAttestationVerifier":
        publication = json.loads(_PUBLICATION_PATH.read_text())
        return cls(
            bytes.fromhex(publication["verify_key"]),
            publication["expected_measurement"],
        )

    def measurement(self) -> str:
        return self._measurement

    def verify(self, payload: bytes, signature: str) -> bool:
        try:
            self._verify_key.verify(payload, bytes.fromhex(signature))
            return True
        except Exception:
            return False


class GcpEnclaveSigner(EnclaveSigner):
    """TEE-side signing seam for the future Confidential Space deployment."""

    def __init__(self, *args, **kwargs):  # pragma: no cover - deployment step
        raise NotImplementedError("bind an enclave signing key to the Confidential Space quote")


class GcpAttestationVerifier(AttestationVerifier):
    """Public quote/evidence verification seam for C and R."""

    def __init__(self, *args, **kwargs):  # pragma: no cover - deployment step
        raise NotImplementedError("verify the quote and expected measurement against vendor roots")


# ---------------------------------------------------------------------------
# ct/0.1 compatibility, for the grafted `ct_core.harness` / `ct_core.roles` path
# used by the conflict (RQ2) case.
#
# That branch models attestation as ONE object that both signs and verifies, where
# ct/0.2 splits it into `EnclaveSigner` (holds the secret) and `AttestationVerifier`
# (public material only) precisely so a verifier cannot forge a certificate. The split
# is the stronger design and stays the default; this class exists so the RQ2 harness
# runs unmodified, and it is deliberately built on the same seed and the same
# measurement function as the split classes, so both paths attest to the same package.
# ---------------------------------------------------------------------------


class Attestor:
    """Interface for the trusted-hardware attestation root (ct/0.1 shape)."""

    def measurement(self) -> str:
        raise NotImplementedError

    def sign(self, payload: bytes) -> str:
        raise NotImplementedError

    @staticmethod
    def verify(payload: bytes, sig_hex: str) -> bool:
        raise NotImplementedError


class StubAttestor(Attestor):
    """Local stand-in for the TEE attestation root (Ed25519 manufacturer key)."""

    def __init__(self) -> None:
        self._signing_key = _create_signing_key()
        self._measurement = _workload_measurement()

    def measurement(self) -> str:
        return self._measurement

    def sign(self, payload: bytes) -> str:
        return self._signing_key.sign(payload).signature.hex()

    @staticmethod
    def verify(payload: bytes, sig_hex: str) -> bool:
        try:
            _create_signing_key().verify_key.verify(payload, bytes.fromhex(sig_hex))
            return True
        except Exception:
            return False
