"""Prover: the untrusted host that relays bytes between verifier and enclave.

The honest prover is a transparent byte pipe; it only ever sees ciphertext and
signatures, never the queries, labels, responses, or loss (those are sealed to
keys it does not hold). The prover/host is exactly the untrusted relay in the
threat model, so the protocol's attack demos live here.

`--tamper-round K`: the interesting attack. The prover CAN seal a query of its own
choosing to the enclave's public channel key pk_ch (sealing is anonymous), so at
round K it replaces the verifier's ciphertext with a re-encryption of a different
query. The enclave decrypts and runs it -- but the per-round digest t_K it echoes
is over the ciphertext the ENCLAVE received, which no longer matches what the
verifier sent, so the verifier aborts at round K. This is precisely why integrity
must be ciphertext-level, not from the sealing.
"""

from __future__ import annotations

import argparse
import socket
import threading

from ..config import ENCLAVE_PORT, PROVER_PORT
from ..protocol import sealing, wire

# Shared between the two relay directions: pk_ch is learned from the enclave's
# commit (downstream) and used to forge a query on tamper (upstream).
_state: dict = {}


def _relay_down(src: socket.socket, dst: socket.socket) -> None:
    """Enclave -> verifier: pass through, but capture pk_ch from the commit."""
    reader = wire.LineReader(src)
    while True:
        msg, raw = reader.recv()
        if msg is None:
            break
        if msg.get("type") == wire.COMMIT and "pk_ch" in msg:
            _state["pk_ch"] = bytes.fromhex(msg["pk_ch"])
        dst.sendall(raw)
    _close_write(dst)


def _relay_up(src: socket.socket, dst: socket.socket, tamper_round: int | None) -> None:
    """Verifier -> enclave: pass through, optionally re-sealing the round-K query."""
    reader = wire.LineReader(src)
    while True:
        msg, raw = reader.recv()
        if msg is None:
            break
        if (tamper_round is not None and msg.get("type") == wire.QUERY
                and msg.get("i") == tamper_round and "pk_ch" in _state):
            forged = sealing.seal_json(_state["pk_ch"],
                                       {"i": tamper_round, "x": "SUBSTITUTED BY PROVER"})
            raw = wire.encode({"type": wire.QUERY, "i": tamper_round, "c": forged}) + b"\n"
            print(f"[prover] re-sealed query at round {tamper_round}", flush=True)
        dst.sendall(raw)
    _close_write(dst)


def _close_write(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_WR)
    except OSError:
        pass


def run_prover(listen_port: int, enclave_port: int, tamper_round: int | None = None) -> None:
    srv = socket.create_server(("127.0.0.1", listen_port))
    print(f"[prover] listening on {listen_port} -> enclave {enclave_port}"
          + (f" (tamper round {tamper_round})" if tamper_round else ""), flush=True)
    client, _ = srv.accept()
    upstream = wire.dial(enclave_port)
    down = threading.Thread(target=_relay_down, args=(upstream, client))
    up = threading.Thread(target=_relay_up, args=(client, upstream, tamper_round))
    down.start(); up.start(); down.join(); up.join()
    client.close(); upstream.close(); srv.close()
    print("[prover] done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Certified-taps prover (relay + attack hooks).")
    ap.add_argument("--listen", type=int, default=PROVER_PORT)
    ap.add_argument("--enclave", type=int, default=ENCLAVE_PORT)
    ap.add_argument("--tamper-round", type=int, default=None,
                    help="re-seal the query at this round (attack demo)")
    args = ap.parse_args()
    run_prover(args.listen, args.enclave, args.tamper_round)


if __name__ == "__main__":
    main()
