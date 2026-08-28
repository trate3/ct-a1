"""Run one certified session in-process.

Spins the enclave (serve_session) in a thread and the verifier in the caller,
connected over a loopback socket, and returns the certificate. This is the shared
in-process pattern used by the eval runners (gaming demo, real-model slice) and by
the tests, so they don't each re-implement the socket plumbing. For the real
multi-process topology, run the roles as separate `python -m` processes instead.
"""

from __future__ import annotations

import socket
import threading
from dataclasses import dataclass

from ..agent import Agent
from ..interfaces import Monitor, TestEnvironment
from ..protocol.attestor import Attestor
from ..protocol.wire import dial
from ..roles.enclave import serve_session
from ..roles.verifier import run_verifier


@dataclass
class SessionTelemetry:
    """Local evaluation telemetry; never included in a certificate."""

    sent_bytes: int = 0
    received_bytes: int = 0


class _CountingSocket:
    """Count bytes at the verifier side without changing protocol code."""

    def __init__(self, sock: socket.socket, telemetry: SessionTelemetry):
        self._sock = sock
        self._telemetry = telemetry

    def sendall(self, data: bytes) -> None:
        self._sock.sendall(data)
        self._telemetry.sent_bytes += len(data)

    def recv(self, size: int) -> bytes:
        data = self._sock.recv(size)
        self._telemetry.received_bytes += len(data)
        return data


def run_session(
    agent: Agent,
    tap: Monitor,
    env: TestEnvironment,
    attestor: Attestor | None = None,
    progress=None,
    commitment_randomizer: bytes | None = None,
    dependency_certificates: tuple[dict, ...] = (),
    telemetry: SessionTelemetry | None = None,
) -> dict:
    """Certify `tap` over `env` with the committed `agent`; return the certificate dict.

    `progress` is forwarded to the verifier and called `progress(round_i, n)` at the
    start of each round (useful when a real model makes each round slow).
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    def serve() -> None:
        srv = socket.create_server(("127.0.0.1", port))
        conn, _ = srv.accept()
        with conn:
            serve_session(conn, agent, tap, env.n(), attestor,
                          commitment_randomizer=commitment_randomizer,
                          dependency_certificates=dependency_certificates)
        srv.close()

    threading.Thread(target=serve, daemon=True).start()
    sock = dial(port)
    try:
        measured_sock = _CountingSocket(sock, telemetry) if telemetry else sock
        return run_verifier(measured_sock, env, attestor, progress=progress,
                            dependency_certificates=dependency_certificates)
    finally:
        sock.close()
