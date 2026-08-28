"""Wire protocol: JSON-lines messages over a byte stream.

One message type per step of the protocol flow. Queries, labels, responses, and the
certificate payload travel sealed (protocol/sealing.py), so the relaying prover
carries only ciphertext it cannot open. `c` is a query/label ciphertext sealed to the
enclave's pk_ch; `d` is a response/certout ciphertext sealed to the verifier's pk_B.

  commit    enclave -> verifier   version, m_ref, N, h_c, pk_ch, n_rounds,
                                  dependency certificate digests, signature   (attested)
  query     verifier -> enclave   round i, c = seal(pk_ch, {i, x, pk_B on round 1})
  response  enclave -> verifier   round i, d = seal(pk_B, {i, y, t_i}), t_i, signature
  certify   verifier -> enclave   c = seal(pk_ch, {labels})
  certout   enclave -> verifier   d = seal(pk_B, {version, n_rounds,
                                  h_T, t_cert, m, m_fp}), signature

No extra phase is needed for certificate chaining.  The commit and final signature
bind the digest of each exact upstream certificate; the verifier supplies and checks
those certificates before revealing its first query.

Localhost sockets stand in for a real network hop, with the prover sitting on that
hop and relaying each message verbatim. `send` returns the exact bytes written and
`LineReader.recv` returns (message, exact raw bytes), which is how the prover forwards
a message without opening it. The transcript digests run over the sealed ciphertext
bytes, not these envelopes (protocol/digests.py).
"""

from __future__ import annotations

import json
import socket
import time

# Message type tags.
COMMIT = "commit"
QUERY = "query"
RESPONSE = "response"
CERTIFY = "certify"
CERTOUT = "certout"

_ENCODING = "utf-8"


def dial(port: int, host: str = "127.0.0.1", timeout: float = 10.0) -> socket.socket:
    """Connect to a peer, retrying while it is still coming up.

    Each process accepts exactly one connection, so callers must not open a
    throwaway probe connection; this retry tolerates the startup race instead.
    """
    deadline = time.time() + timeout
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=1.0)
        except OSError:
            if time.time() >= deadline:
                raise
            time.sleep(0.05)
            continue
        # The 1s above is the per-attempt CONNECT timeout. Clear it before returning:
        # a round's response waits on real model inference, which far outlasts 1s, so
        # reads must block. A dead peer closes the socket and recv returns cleanly.
        sock.settimeout(None)
        return sock


def encode(obj: dict) -> bytes:
    """Canonical, deterministic serialization of one message (no trailing newline)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode(_ENCODING)


def send(sock: socket.socket, obj: dict) -> bytes:
    """Send one message as a JSON line. Returns the exact bytes sent."""
    data = encode(obj) + b"\n"
    sock.sendall(data)
    return data


class LineReader:
    """Buffered newline-delimited reader; yields (parsed message, exact raw bytes)."""

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._buf = b""

    def recv(self) -> tuple[dict, bytes] | tuple[None, None]:
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                return None, None  # peer closed
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return json.loads(line.decode(_ENCODING)), line + b"\n"
