"""Third-party certificate verification.

Anyone holding the manufacturer public key can check a published certificate, and
(if the reveal is published) that the opened tests match the committed h_T. This
is the "any third party" role of the protocol.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from ..config import PROTOCOL_VERSION
from ..protocol.attestor import Attestor, StubAttestor
from ..protocol.digests import certificate_digest, signing_payload, tests_digest


def _signature_valid(cert: dict, attestor: Attestor) -> bool:
    try:
        if (cert["version"] != PROTOCOL_VERSION
                or isinstance(cert["n_rounds"], bool)
                or not isinstance(cert["n_rounds"], int)
                or cert["n_rounds"] <= 0):
            return False
        dependencies = cert.get("dependencies", [])
        if (not isinstance(dependencies, list)
                or any(not isinstance(d, str) for d in dependencies)
                or len(set(dependencies)) != len(dependencies)):
            return False
        return attestor.verify(
            signing_payload("certify", cert["version"], cert["n_rounds"],
                            cert["m_ref"], cert["N"], cert["h_c"], cert["h_T"],
                            cert["t_cert"], cert["m"], cert["m_fp"], dependencies),
            cert["sig"])
    except (KeyError, TypeError, ValueError):
        return False


def certificate_valid(
    cert: dict,
    referenced_certificates: tuple[dict, ...] = (),
    attestor: Attestor | None = None,
) -> bool:
    """Verify a certificate and every exact certificate it names as a dependency.

    Dependency digests include the upstream signature.  A chained certificate fails
    closed if an upstream certificate is absent, substituted, malformed, or invalid.
    """
    attestor = attestor or StubAttestor()
    try:
        by_digest = {certificate_digest(ref): ref for ref in referenced_certificates}
    except (KeyError, TypeError, ValueError):
        return False

    def check(current: dict, stack: frozenset[str]) -> bool:
        try:
            current_digest = certificate_digest(current)
        except (KeyError, TypeError, ValueError):
            return False
        if current_digest in stack or not _signature_valid(current, attestor):
            return False
        next_stack = stack | {current_digest}
        for dependency in current.get("dependencies", []):
            upstream = by_digest.get(dependency)
            if upstream is None or not check(upstream, next_stack):
                return False
        return True

    return check(cert, frozenset())


def verify_certificate(
    cert: dict,
    reveal: dict | None = None,
    referenced_certificates: tuple[dict, ...] = (),
    attestor: Attestor | None = None,
) -> bool:
    ok = certificate_valid(cert, referenced_certificates, attestor)
    print(f"certificate chain     : {'VALID' if ok else 'INVALID'}")
    try:
        print(f"  commitment h_c      : {cert['h_c'][:16]}...")
        print(f"  session N           : {cert['N'][:8]}...")
        print(f"  released m(FN)/m_fp : {cert['m']:.3f} / {cert['m_fp']:.3f}")
    except (KeyError, TypeError, ValueError):
        return False
    if cert.get("dependencies"):
        print(f"  dependencies        : {len(cert['dependencies'])} referenced")
    if reveal is not None:
        try:
            tests = [tuple(t) for t in reveal["tests"]]
            match = tests_digest(tests) == cert["h_T"]
        except (KeyError, TypeError, ValueError):
            match = False
        print(f"  revealed T vs h_T   : {'MATCH' if match else 'MISMATCH'}")
        ok = ok and match
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify a Certified Taps certificate.")
    ap.add_argument("cert", help="path to cert.json")
    ap.add_argument("reveal", nargs="?", help="optional path to reveal.json")
    ap.add_argument("--dependency", action="append", default=[],
                    help="referenced certificate JSON (repeat for nested dependencies)")
    args = ap.parse_args()
    cert = json.loads(pathlib.Path(args.cert).read_text())
    reveal = json.loads(pathlib.Path(args.reveal).read_text()) if args.reveal else None
    dependencies = tuple(json.loads(pathlib.Path(path).read_text())
                         for path in args.dependency)
    raise SystemExit(0 if verify_certificate(cert, reveal, dependencies) else 1)


if __name__ == "__main__":
    main()
