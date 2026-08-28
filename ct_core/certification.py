"""Certified-taps protocol and the complete trusted-workload entry point."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import pathlib
import secrets
import socket
import time
import threading
from dataclasses import dataclass, field

from .agent import Agent
from .config import ENV_PLAINTEXT_BYTES, NONCE_BYTES, PROTOCOL_VERSION, RANDOMIZER_BYTES
from .interfaces import (
    AgentDriver,
    ArtifactPackage,
    Monitor,
    MonitorInput,
    TapDriver,
    TestEnvironment,
)
from .protocol import sealing
from .protocol.attestor import (
    AttestationVerifier,
    EnclaveSigner,
    StubAttestationVerifier,
    StubEnclaveSigner,
    initialize_local_attestation,
)
from .protocol.digests import (
    artifact_commitment_digest,
    sha256_hex,
    signing_payload,
)

COMMIT, TEST, CERTOUT = "commit", "test", "certout"
SPEC_VERSION, ENV_VERSION, COMPONENT_VERSION = (
    "ct-spec/2", "ct-environment/1", "ct-components/1")
SCHEMAS = {"query": "utf8", "response": "utf8", "prediction": "bit", "label": "bit"}


class CertifierAbort(Exception):
    pass


@dataclass
class SessionTelemetry:
    sent_bytes: int = 0
    received_bytes: int = 0
    # Bytes attributed to each protocol message type. The session is three messages
    # -- commit, test, certout -- and the test message carries the sealed environment
    # padded to a fixed length, so it dominates the total and is constant by design.
    # Reporting the total alone makes the protocol look expensive in proportion to a
    # deliberate size-hiding property; splitting it separates delivering E from
    # running the protocol around it.
    by_message: dict = field(default_factory=dict)

    def note(self, kind: str | None, direction: str, count: int) -> None:
        entry = self.by_message.setdefault(kind or "unknown",
                                           {"sent_bytes": 0, "received_bytes": 0})
        entry[f"{direction}_bytes"] += count


class _CountingSocket:
    def __init__(self, sock, telemetry):
        self.sock, self.telemetry = sock, telemetry

    def sendall(self, data):
        self.sock.sendall(data)
        self.telemetry.sent_bytes += len(data)

    def recv(self, size):
        data = self.sock.recv(size)
        self.telemetry.received_bytes += len(data)
        return data


def _canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _account(sock, kind: str | None, direction: str, count: int) -> None:
    telemetry = getattr(sock, "telemetry", None)
    if telemetry is not None:
        telemetry.note(kind, direction, count)


def _send(sock, message: dict) -> None:
    payload = _canonical(message) + b"\n"
    sock.sendall(payload)
    _account(sock, message.get("type"), "sent", len(payload))


def _receive(sock) -> dict | None:
    raw = b""
    while not raw.endswith(b"\n"):
        chunk = sock.recv(65536)
        if not chunk:
            return None
        raw += chunk
    value = json.loads(raw)
    if not isinstance(value, dict):
        return None
    _account(sock, value.get("type"), "received", len(raw))
    return value


def _driver_artifacts(driver) -> dict[str, str]:
    module_path = inspect.getsourcefile(driver)
    if not module_path:
        raise ValueError("environment driver must have source")
    paths = (module_path, *driver.code_paths())
    artifacts = {}
    for index, value in enumerate(paths):
        path = pathlib.Path(value)
        key = f"{index}:{path.name}"
        artifacts[key] = base64.b64encode(path.read_bytes()).decode()
    return artifacts


def _registry(drivers, kind: str) -> dict[str, type]:
    registry = {}
    for driver in drivers:
        identifier = getattr(driver, "DRIVER", "")
        if not isinstance(identifier, str) or not identifier or identifier in registry:
            raise ValueError(f"{kind} drivers must have unique nonempty identifiers")
        registry[identifier] = driver
    return registry


def _file_digest(path: pathlib.Path) -> str:
    """Hash a provisioned artifact without copying it into Python memory."""
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(size.to_bytes(8, "big"))
    with path.open("rb") as artifact:
        while chunk := artifact.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _component_record(kind, package, driver):
    if not isinstance(package, ArtifactPackage) or not isinstance(package.config, dict):
        raise ValueError(f"malformed {kind} package")
    config = json.loads(_canonical(package.config))
    contents = {}
    artifacts = {}
    for item in package.artifacts:
        if (not isinstance(item, tuple) or len(item) != 2
                or not isinstance(item[0], str) or not item[0]
                or not isinstance(item[1], (str, bytes)) or not item[1]):
            raise ValueError(f"malformed {kind} artifact")
        name, content = item
        if name in contents:
            raise ValueError(f"duplicate {kind} artifact name")
        if isinstance(content, bytes):
            contents[name] = content
            artifacts[name] = {"bytes": len(content), "sha256": sha256_hex(content)}
        else:
            path = pathlib.Path(content)
            if not path.is_file():
                raise ValueError(f"missing {kind} artifact: {name}")
            contents[name] = str(path)
            artifacts[name] = {"bytes": path.stat().st_size,
                               "sha256": _file_digest(path)}
    code = {
        name: sha256_hex(base64.b64decode(value))
        for name, value in _driver_artifacts(driver).items()
    }
    return ({"version": COMPONENT_VERSION, "kind": kind, "driver": package.driver,
             "code": code, "config": config, "artifacts": artifacts}, contents)


@dataclass(frozen=True)
class PreparedComponents:
    """Registered loaders and the exact canonical object committed as (Pi, g)."""

    encoded: bytes
    agent_driver: AgentDriver
    tap_driver: TapDriver


def prepare_components(
    agent_package: ArtifactPackage,
    tap_package: ArtifactPackage,
    agent_drivers: tuple[type[AgentDriver], ...],
    tap_drivers: tuple[type[TapDriver], ...],
) -> PreparedComponents:
    """Hash immutable provider artifacts, then invoke registered trusted loaders."""
    if not isinstance(agent_package, ArtifactPackage):
        raise ValueError("malformed agent package")
    if not isinstance(tap_package, ArtifactPackage):
        raise ValueError("malformed tap package")
    agent_type = _registry(agent_drivers, "agent").get(agent_package.driver)
    tap_type = _registry(tap_drivers, "tap").get(tap_package.driver)
    if agent_type is None or tap_type is None:
        raise ValueError("component driver is not registered by this workload")
    agent_record, agent_artifacts = _component_record("agent", agent_package, agent_type)
    tap_record, tap_artifacts = _component_record("tap", tap_package, tap_type)
    agent_driver = agent_type.from_package(agent_record["config"], agent_artifacts)
    tap_driver = tap_type.from_package(tap_record["config"], tap_artifacts)
    # Detect a file changed while its registered loader consumed it.
    if (_component_record("agent", agent_package, agent_type)[0] != agent_record
            or _component_record("tap", tap_package, tap_type)[0] != tap_record):
        raise ValueError("component artifact changed while it was loaded")
    encoded = _canonical({"version": COMPONENT_VERSION,
                          "agent": agent_record, "tap": tap_record})
    return PreparedComponents(encoded, agent_driver, tap_driver)


def _environment_package(environment: TestEnvironment) -> tuple[bytes, type[TestEnvironment]]:
    driver = environment.runtime_driver()
    if not driver.DRIVER:
        raise ValueError("environment driver must declare DRIVER")
    package = {
        "version": ENV_VERSION,
        "driver": driver.DRIVER,
        "artifacts": _driver_artifacts(driver),
        "config": environment.package_config(),
    }
    encoded = _canonical(package)
    if len(encoded) > ENV_PLAINTEXT_BYTES - 4:
        raise ValueError("environment package exceeds public len_E")
    return encoded, driver


def prepare_environment(environment: TestEnvironment) -> tuple[dict, bytes]:
    """Build the public spec and fixed-length inert E plaintext on C's side."""
    encoded, _ = _environment_package(environment)
    spec = {
        "version": SPEC_VERSION,
        "n_trials": environment.trial_count(),
        "max_steps": environment.max_steps(),
        "schemas": SCHEMAS,
        "aggregation": "any-step/v1",
        "release": "fn-fp/v1",
        "id_E": sha256_hex(encoded),
        "len_E": ENV_PLAINTEXT_BYTES,
    }
    _validate_spec(spec)
    return spec, _pad_environment(encoded)


def spec_digest(spec: dict) -> str:
    return sha256_hex(_canonical(spec))


def _validate_spec(spec: dict) -> None:
    if (set(spec) != {"version", "n_trials", "max_steps", "schemas", "aggregation",
                     "release", "id_E", "len_E"}
            or spec["version"] != SPEC_VERSION
            or isinstance(spec["n_trials"], bool) or not isinstance(spec["n_trials"], int)
            or spec["n_trials"] <= 0
            or isinstance(spec["max_steps"], bool) or not isinstance(spec["max_steps"], int)
            or spec["max_steps"] <= 0
            or spec["schemas"] != SCHEMAS
            or spec["aggregation"] != "any-step/v1"
            or spec["release"] != "fn-fp/v1"
            or not isinstance(spec["id_E"], str) or len(spec["id_E"]) != 64
            or any(char not in "0123456789abcdef" for char in spec["id_E"])
            or spec["len_E"] != ENV_PLAINTEXT_BYTES):
        raise ValueError("unsupported or malformed measurement specification")


def _pad_environment(encoded: bytes) -> bytes:
    if len(encoded) > ENV_PLAINTEXT_BYTES - 4:
        raise ValueError("environment package exceeds public len_E")
    return len(encoded).to_bytes(4, "big") + encoded + bytes(ENV_PLAINTEXT_BYTES - 4 - len(encoded))


def _unpad_environment(padded: bytes) -> bytes:
    if len(padded) != ENV_PLAINTEXT_BYTES:
        raise ValueError("environment plaintext has wrong length")
    length = int.from_bytes(padded[:4], "big")
    if length > ENV_PLAINTEXT_BYTES - 4 or any(padded[4 + length:]):
        raise ValueError("environment padding is malformed")
    return padded[4:4 + length]


def _load_environment(encoded: bytes, spec: dict,
                      drivers: tuple[type[TestEnvironment], ...]) -> tuple[type[TestEnvironment], dict]:
    """Authenticate inert bytes and registered source before constructing E."""
    _validate_spec(spec)
    if sha256_hex(encoded) != spec["id_E"]:
        raise ValueError("environment digest mismatch")
    package = json.loads(encoded)
    if (not isinstance(package, dict)
            or set(package) != {"version", "driver", "artifacts", "config"}
            or package["version"] != ENV_VERSION
            or not isinstance(package["config"], dict)
            or not isinstance(package["artifacts"], dict)):
        raise ValueError("malformed environment package")
    registry = {driver.DRIVER: driver for driver in drivers}
    driver = registry.get(package["driver"])
    if driver is None or package["artifacts"] != _driver_artifacts(driver):
        raise ValueError("environment code is not registered by this workload")
    config = package["config"]
    environment = driver.from_config(json.loads(_canonical(config)))
    if (environment.trial_count() != spec["n_trials"]
            or environment.max_steps() != spec["max_steps"]):
        raise ValueError("environment configuration does not match specification")
    return driver, config


def _commit_payload(message):
    return signing_payload("commit", message["version"], message["m_ref"], message["N"],
                           message["h_c"], message["h_s"], message["pk_ch"])


def _certificate_payload(cert):
    return signing_payload("certify", cert["version"], cert["m_ref"], cert["N"],
                           cert["h_c"], cert["h_s"], cert["t_E"], cert["m"],
                           cert["m_fp"])


def measure(components: PreparedComponents, padded_environment: bytes, spec: dict,
            drivers: tuple[type[TestEnvironment], ...], progress=None,
            entropy=secrets.token_bytes) -> dict:
    """Execute E behind the registered driver interface over independent trials."""
    encoded_environment = _unpad_environment(padded_environment)
    driver, config = _load_environment(encoded_environment, spec, drivers)
    records, trial_predictions, trial_labels = [], [], []
    agent_sessions, tap_sessions = [], []

    for trial_index in range(1, spec["n_trials"] + 1):
        environment = driver.from_config(json.loads(_canonical(config)))
        agent = components.agent_driver.new_agent(entropy(32))
        tap = components.tap_driver.new_tap(entropy(32))
        if (not isinstance(agent, Agent) or not isinstance(tap, Monitor)
                or any(agent is prior for prior in agent_sessions)
                or any(tap is prior for prior in tap_sessions)):
            raise ValueError("registered drivers must construct fresh agent and tap sessions")
        agent_sessions.append(agent)
        tap_sessions.append(tap)
        capture = tap.probe_layers if getattr(tap, "needs_activations", False) else None
        trial = environment.new_trial(trial_index, entropy(32))
        transcript: list[tuple[str, str]] = []
        predictions = []
        if progress:
            progress(trial_index, spec["n_trials"])
        for step in range(1, spec["max_steps"] + 2):
            query = trial.next_query(tuple(transcript))
            if query is None:
                break
            if step > spec["max_steps"] or not isinstance(query, str):
                raise ValueError("environment exceeded its step bound or emitted a non-string")
            generation = agent.generate(query, capture_layers=capture)
            if not isinstance(generation.answer, str):
                raise ValueError("program response must be a string")
            prediction = tap.predict(MonitorInput(
                x=query, answer=generation.answer, thought=generation.thought,
                activations=generation.activations,
            ))
            if prediction.z_hat not in (0, 1):
                raise ValueError("tap output must be binary")
            transcript.append((query, generation.answer))
            predictions.append(prediction.z_hat)
            records.append({"trial": trial_index, "step": step, "x": query,
                            "y": generation.answer, "z_hat": prediction.z_hat})
        if not transcript or not predictions:
            raise ValueError("environment trial must complete at least one step")
        label = trial.score(tuple(transcript))
        if label not in (0, 1):
            raise ValueError("environment label must be binary")
        trial_predictions.append(max(predictions))
        trial_labels.append(label)
        for record in records:
            if record["trial"] == trial_index:
                record["z"] = label

    n = spec["n_trials"]
    return {
        "m": sum(z and not z_hat for z_hat, z in zip(trial_predictions, trial_labels)) / n,
        "m_fp": sum(not z and z_hat for z_hat, z in zip(trial_predictions, trial_labels)) / n,
        "_tests": [(record["x"], record["z"]) for record in records],
        "_steps": records,
        "_trial_predictions": trial_predictions,
        "_trial_labels": trial_labels,
    }


def measure_loaded(build_agent, build_tap, padded_environment: bytes, spec: dict,
                   drivers: tuple[type[TestEnvironment], ...]) -> dict:
    """Offline analysis on already-loaded objects; not a certification boundary."""
    class LoadedAgent:
        def new_agent(self, randomness):
            return build_agent()

    class LoadedTap:
        def new_tap(self, randomness):
            return build_tap()

    return measure(
        PreparedComponents(b"", LoadedAgent(), LoadedTap()),
        padded_environment, spec, drivers)


def serve_session(
    sock,
    agent_package: ArtifactPackage,
    tap_package: ArtifactPackage,
    spec: dict,
    *,
    agent_drivers: tuple[type[AgentDriver], ...],
    tap_drivers: tuple[type[TapDriver], ...],
    environment_drivers: tuple[type[TestEnvironment], ...],
    signer: EnclaveSigner | None = None,
    commitment_randomizer: bytes | None = None,
    progress=None,
    phases: dict | None = None,
) -> None:
    """Trusted-workload endpoint: one commit, one E, one public certificate.

    `phases`, like `telemetry` on run_session, is evaluation instrumentation and has
    no effect on the protocol. It exists because the wall time of a session is not one
    quantity: committing the provider's artifacts means hashing every provisioned file,
    which for a multi-gigabyte checkpoint dominates everything else and is entirely
    unrelated to the per-trial cost. Reporting one total conflates the two.
    """
    def mark(name, started):
        if phases is not None:
            phases[name] = time.perf_counter() - started

    try:
        _validate_spec(spec)
        started = time.perf_counter()
        components = prepare_components(
            agent_package, tap_package, agent_drivers, tap_drivers)
        mark("commitment_seconds", started)
    except (KeyError, TypeError, ValueError):
        return
    signer = signer or StubEnclaveSigner()
    randomizer = (secrets.token_bytes(RANDOMIZER_BYTES)
                  if commitment_randomizer is None else commitment_randomizer)
    if not isinstance(randomizer, bytes) or len(randomizer) != RANDOMIZER_BYTES:
        raise ValueError(f"commitment randomizer must be {RANDOMIZER_BYTES} bytes")
    m_ref, nonce = signer.measurement(), secrets.token_hex(NONCE_BYTES)
    h_c = artifact_commitment_digest(components.encoded, randomizer)
    secret_key, public_key = sealing.new_keypair()
    commit = {"type": COMMIT, "version": PROTOCOL_VERSION, "m_ref": m_ref,
              "N": nonce, "h_c": h_c, "h_s": spec_digest(spec),
              "pk_ch": public_key.hex()}
    commit["sig"] = signer.sign(_commit_payload(commit))
    _send(sock, commit)

    try:
        message = _receive(sock)
        if (not message or message.get("type") != TEST
                or not isinstance(message.get("c_E"), str)):
            return
        padded = sealing.unseal_bytes(secret_key, message["c_E"])
        started = time.perf_counter()
        result = measure(components, padded, spec, environment_drivers, progress)
        mark("model_run_seconds", started)
    except Exception:
        return

    cert = {"version": PROTOCOL_VERSION, "m_ref": m_ref, "N": nonce,
            "h_c": h_c, "h_s": spec_digest(spec),
            "t_E": sha256_hex(sealing.ciphertext_bytes(message["c_E"])),
            "m": result["m"], "m_fp": result["m_fp"]}
    cert["sig"] = signer.sign(_certificate_payload(cert))
    _send(sock, {"type": CERTOUT, **cert})


def run_certifier(
    sock,
    environment: TestEnvironment,
    spec=None,
    verifier: AttestationVerifier | None = None,
) -> dict:
    """Certifier endpoint; C packages E, while all trial execution stays remote."""
    expected_spec, padded = prepare_environment(environment)
    spec = expected_spec if spec is None else spec
    if spec != expected_spec:
        raise CertifierAbort("environment does not match specification")
    try:
        commit = _receive(sock)
        verifier = verifier or StubAttestationVerifier.from_publication()
        if (not commit or commit.get("type") != COMMIT
                or commit.get("version") != PROTOCOL_VERSION
                or commit.get("m_ref") != verifier.measurement()
                or commit.get("h_s") != spec_digest(spec)
                or not verifier.verify(_commit_payload(commit), commit.get("sig", ""))):
            raise CertifierAbort("commit verification failed")
        ciphertext = sealing.seal_bytes(bytes.fromhex(commit["pk_ch"]), padded)
    except CertifierAbort:
        raise
    except Exception as error:
        raise CertifierAbort("malformed commit") from error

    _send(sock, {"type": TEST, "c_E": ciphertext})
    output = _receive(sock)
    if not output or output.get("type") != CERTOUT:
        raise CertifierAbort("no certification output")
    cert = {key: value for key, value in output.items() if key != "type"}
    if (any(cert.get(key) != commit[key] for key in ("version", "m_ref", "N", "h_c", "h_s"))
            or cert.get("t_E") != sha256_hex(sealing.ciphertext_bytes(ciphertext))):
        raise CertifierAbort("certificate is not bound to this environment message")
    if not certificate_valid(cert, verifier, spec):
        raise CertifierAbort("certificate verification failed")
    return cert


def certificate_valid(cert, verifier=None, spec=None) -> bool:
    try:
        verifier = verifier or StubAttestationVerifier.from_publication()
        return (cert["version"] == PROTOCOL_VERSION
                and cert["m_ref"] == verifier.measurement()
                and (spec is None or cert["h_s"] == spec_digest(spec))
                and 0 <= cert["m"] <= 1 and 0 <= cert["m_fp"] <= 1
                and verifier.verify(_certificate_payload(cert), cert["sig"]))
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False


def verify_certificate(cert, verifier=None, spec=None) -> bool:
    valid = certificate_valid(cert, verifier, spec)
    print(f"certificate           : {'VALID' if valid else 'INVALID'}")
    if valid:
        print(f"  commitment h_c      : {cert['h_c'][:16]}...")
        print(f"  specification h_s   : {cert['h_s'][:16]}...")
        print(f"  released m(FN)/m_fp : {cert['m']:.3f} / {cert['m_fp']:.3f}")
    return valid


def run_session(
    agent_package: ArtifactPackage,
    tap_package: ArtifactPackage,
    environment: TestEnvironment,
    *,
    agent_drivers: tuple[type[AgentDriver], ...],
    tap_drivers: tuple[type[TapDriver], ...],
    signer=None,
    verifier=None,
    progress=None,
    commitment_randomizer=None,
    telemetry=None,
    phases=None,
):
    """Local topology: the same workload and certifier endpoints over a socket pair."""
    if signer is None and verifier is None:
        verifier = initialize_local_attestation()
        signer = StubEnclaveSigner()
    elif signer is None or verifier is None:
        raise ValueError("local sessions require a separate signer and verifier")
    spec, _ = prepare_environment(environment)
    environment_driver = environment.runtime_driver()
    workload_sock, certifier_sock = socket.socketpair()

    def serve():
        with workload_sock:
            serve_session(
                workload_sock,
                agent_package,
                tap_package,
                spec,
                agent_drivers=agent_drivers,
                tap_drivers=tap_drivers,
                environment_drivers=(environment_driver,),
                signer=signer,
                commitment_randomizer=commitment_randomizer,
                progress=progress,
                phases=phases,
            )

    thread = threading.Thread(target=serve)
    thread.start()
    measured = _CountingSocket(certifier_sock, telemetry) if telemetry else certifier_sock
    try:
        return run_certifier(measured, environment, spec, verifier)
    finally:
        certifier_sock.close()
        thread.join()
