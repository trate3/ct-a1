"""Separate-process rehearsal of the three-message certified-taps protocol."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from ct_core.certification import (
    COMMIT, TEST, CertifierAbort, prepare_environment, run_certifier,
    serve_session, verify_certificate,
)
from ct_core.config import ENCLAVE_PORT, RELAY_PORT
from ct_core.protocol import sealing
from ct_core.protocol.attestor import initialize_local_attestation
from evals.basic_mocked import FIXTURE
from evals.basic_mocked.mocked_protocol_components import (
    MockAgentDriver, MockTapDriver, agent_package as mock_agent_package,
    build_agent as mock_agent, build_comparator,
    build_environment as mock_environment, build_tap as mock_tap,
    tap_package as mock_tap_package,
)
from evals_runner.util import run_tap_quality, write_certificate

REPO = Path(__file__).resolve().parents[1]


def _case_args(parser):
    parser.add_argument("--case", choices=("mock", "confidence", "financial"), default="mock")
    parser.add_argument("--data", default=str(FIXTURE))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--program-model", default="google/gemma-4-E2B-it")
    parser.add_argument("--monitor-model", default=None)
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--detector", default=None)
    parser.add_argument("--n-trajectories", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=None)


def configure_parser(parser):
    commands = parser.add_subparsers(dest="service", required=True)
    for name, run in (("attestation-setup", run_attestation_setup),
                      ("e2e", run_e2e), ("spec", run_spec), ("workload", run_workload),
                      ("certifier", run_certifier_service), ("relay", run_relay),
                      ("verify", run_verify)):
        command = commands.add_parser(name)
        if name in ("e2e", "spec", "workload", "certifier"):
            _case_args(command)
        if name == "e2e":
            command.add_argument("--tamper-environment", action="store_true")
            command.add_argument("--out", default=None)
        elif name in ("spec", "workload"):
            command.add_argument("--spec", required=True)
        elif name == "certifier":
            command.add_argument("--spec", required=True)
            command.add_argument("--peer-host", default="127.0.0.1")
            command.add_argument("--peer-port", type=int, default=RELAY_PORT)
            command.add_argument("--out", default=None)
        elif name == "relay":
            command.add_argument("--listen-host", default="127.0.0.1")
            command.add_argument("--listen-port", type=int, default=RELAY_PORT)
            command.add_argument("--workload-host", default="127.0.0.1")
            command.add_argument("--workload-port", type=int, default=ENCLAVE_PORT)
            command.add_argument("--replace-environment")
        elif name == "verify":
            command.add_argument("cert")
            command.add_argument("--spec", required=True)
        if name == "workload":
            command.add_argument("--listen-host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=ENCLAVE_PORT)
        command.set_defaults(run=run)


def _components(args):
    if args.case == "mock":
        return (mock_agent_package(args.data), mock_tap_package(),
                (MockAgentDriver,), (MockTapDriver,))
    if args.case == "confidence":
        from evals.confidence.cot_confidence_components import (
            ConfidenceAgentDriver, ConfidenceTapDriver, component_packages,
        )
        agent, tap = component_packages(
            args.program_model, args.monitor_model, args.device or "auto", args.dtype,
            args.max_new_tokens or 512, verbose=True)
        return agent, tap, (ConfidenceAgentDriver,), (ConfidenceTapDriver,)
    from evals.financial.insider_trading_probe_components import (
        FinancialAgentDriver, FinancialTapDriver, component_packages,
    )
    if not args.detector or Path(args.detector).suffix != ".npz":
        raise ValueError("financial workload requires a prepacked non-executable .npz detector")
    agent, tap = component_packages(
        Path(args.detector), model=args.model, device=args.device or "cpu",
        dtype=args.dtype, max_new_tokens=args.max_new_tokens or 128,
        threshold=args.threshold, verbose=True)
    return agent, tap, (FinancialAgentDriver,), (FinancialTapDriver,)


def _environment(args):
    if args.case == "mock":
        return mock_environment(args.data, args.n, args.seed)
    if args.case == "confidence":
        from evals.confidence.cot_confidence_components import build_environment
        return build_environment(args.n, args.seed)
    from evals.financial.insider_trading_probe_components import build_environment
    return build_environment(args.n_trajectories, args.seed)


def _environment_driver(args):
    if args.case in ("mock", "confidence"):
        from util.dynamic_test_environment import StaticEnvironment
        return StaticEnvironment
    from evals.financial.insider_trading_env import InsiderTradingEnvironment
    return InsiderTradingEnvironment


def _dial(host, port):
    for _ in range(200):
        try:
            return socket.create_connection((host, port))
        except OSError:
            time.sleep(0.05)
    raise ConnectionError(f"could not connect to port {port}")


def run_spec(args):
    spec, _ = prepare_environment(_environment(args))
    Path(args.spec).write_text(json.dumps(spec, indent=2) + "\n")
    return 0


def run_attestation_setup(args):
    verifier = initialize_local_attestation()
    print(f"[setup] pinned m_ref={verifier.measurement()}")
    return 0


def run_workload(args):
    server = socket.create_server((args.listen_host, args.port))
    connection, _ = server.accept()
    with connection:
        agent, tap, agent_drivers, tap_drivers = _components(args)
        serve_session(
            connection, agent, tap, json.loads(Path(args.spec).read_text()),
            agent_drivers=agent_drivers, tap_drivers=tap_drivers,
            environment_drivers=(_environment_driver(args),))
    server.close()
    return 0


def run_certifier_service(args):
    environment = _environment(args)
    spec = json.loads(Path(args.spec).read_text())
    connection = _dial(args.peer_host, args.peer_port)
    try:
        cert = run_certifier(connection, environment, spec)
    except CertifierAbort as error:
        print(f"[certifier] ABORT: {error}")
        return 1
    finally:
        connection.close()
    write_certificate(cert, spec, args.out or f"evals_results/{args.case}/session")
    print(f"[certifier] CERTIFIED  m(FN)={cert['m']:.3f}  m_fp={cert['m_fp']:.3f}")
    return 0


def _pipe(source, destination, state, replacement=None):
    for raw in source.makefile("rb"):
        message = json.loads(raw)
        if message.get("type") == COMMIT:
            state["pk"] = bytes.fromhex(message["pk_ch"])
        elif replacement and message.get("type") == TEST:
            ciphertext = sealing.seal_bytes(state["pk"], replacement)
            raw = json.dumps({"type": TEST, "c_E": ciphertext}).encode() + b"\n"
        destination.sendall(raw)
    try:
        destination.shutdown(socket.SHUT_WR)
    except OSError:
        pass


def run_relay(args):
    replacement = (Path(args.replace_environment).read_bytes()
                   if args.replace_environment else None)
    server = socket.create_server((args.listen_host, args.listen_port))
    certifier, _ = server.accept()
    enclave, state = _dial(args.workload_host, args.workload_port), {}
    threads = [threading.Thread(target=_pipe, args=(enclave, certifier, state)),
               threading.Thread(target=_pipe, args=(certifier, enclave, state, replacement))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    certifier.close(); enclave.close(); server.close()
    return 0


def run_verify(args):
    cert = json.loads(Path(args.cert).read_text())
    spec = json.loads(Path(args.spec).read_text())
    return 0 if verify_certificate(cert, spec=spec) else 1


def run_e2e(args):
    initialize_local_attestation()
    environment = _environment(args)
    spec, packed = prepare_environment(environment)
    process_env = {**os.environ, "PYTHONPATH": str(REPO)}
    with tempfile.TemporaryDirectory(prefix="ct-") as temporary:
        spec_path, environment_path = Path(temporary) / "spec.json", Path(temporary) / "E"
        spec_path.write_text(json.dumps(spec)); environment_path.write_bytes(packed)
        component_args = _component_args(args)
        if args.case == "financial":
            from evals.financial.insider_trading_probe_components import latest_detector
            from evals.financial.probe_tap import ApolloProbe
            source = Path(args.detector) if args.detector else latest_detector()
            detector = Path(temporary) / "detector.npz"
            detector.write_bytes(ApolloProbe.load(source).safe_bytes())
            component_args = _component_args(args, detector)
        workload = subprocess.Popen([
            sys.executable, "-m", "evals_runner", "protocol-integrity-exp", "workload",
            "--case", args.case, "--spec", str(spec_path), *component_args], env=process_env)
        relay_command = [sys.executable, "-m", "evals_runner", "protocol-integrity-exp", "relay"]
        if args.tamper_environment:
            relay_command += ["--replace-environment", str(environment_path)]
        relay = subprocess.Popen(relay_command, env=process_env)
        connection, result = _dial("127.0.0.1", RELAY_PORT), 0
        try:
            cert = run_certifier(connection, environment, spec)
            write_certificate(cert, spec, args.out or f"evals_results/{args.case}/session")
            print(f"[certifier] CERTIFIED  m(FN)={cert['m']:.3f}  m_fp={cert['m_fp']:.3f}")
            verify_certificate(cert, spec=spec)
            if args.case == "mock":
                quality = run_tap_quality(lambda: mock_agent(args.data), mock_tap,
                                          mock_environment(args.data, args.n, args.seed),
                                          seed=args.seed,
                                          build_comparator=build_comparator)
                print(f"separation (FN)      : {quality['comparator_minus_tap']['m']:+.3f}")
        except CertifierAbort as error:
            print(f"[certifier] ABORT: {error}")
            result = 1
        finally:
            connection.close()
            for process in (relay, workload):
                process.terminate(); process.wait(timeout=5)
        return result


def _component_args(args, detector=None):
    if args.case == "mock":
        return ["--data", args.data]
    if args.case == "confidence":
        values = ["--program-model", args.program_model, "--dtype", args.dtype]
        if args.monitor_model:
            values += ["--monitor-model", args.monitor_model]
    else:
        values = ["--model", args.model, "--threshold", str(args.threshold)]
        selected = detector or args.detector
        if selected:
            values += ["--detector", str(selected)]
    if args.device:
        values += ["--device", args.device]
    if args.max_new_tokens is not None:
        values += ["--max-new-tokens", str(args.max_new_tokens)]
    return values
