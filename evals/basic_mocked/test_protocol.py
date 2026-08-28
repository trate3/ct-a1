"""GPU-free protocol checks."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from nacl.signing import SigningKey

import ct_core.certification as certification
import ct_core.protocol.attestor as attestation
from ct_core.agent import Agent
from ct_core.certification import (
    certificate_valid, prepare_components, prepare_environment, run_session,
    verify_certificate,
)
from ct_core.config import PROTOCOL_VERSION
from ct_core.protocol.attestor import (
    StubAttestationVerifier, StubEnclaveSigner, initialize_local_attestation,
)
from evals.basic_mocked import FIXTURE
from evals.basic_mocked.mocks import MockBlackBox, MockProgram, MockTap
from evals.basic_mocked.mocked_protocol_components import (
    MockAgentDriver, MockTapDriver, agent_package, tap_package,
)
from evals_runner.util import ExperimentConfig, run_comparison, run_tap_quality, write_result
from util.dynamic_test_environment import StaticEnvironment

DATA, N = str(FIXTURE), 8


def _program():
    return MockProgram.from_jsonl(DATA)


def _env():
    return StaticEnvironment.from_jsonl(DATA, N, seed=7)


def _session(environment=None):
    return run_session(
        agent_package(DATA), tap_package(), environment or _env(),
        agent_drivers=(MockAgentDriver,), tap_drivers=(MockTapDriver,))


def _components():
    return prepare_components(
        agent_package(DATA), tap_package(), (MockAgentDriver,), (MockTapDriver,))


def test_honest_session_is_aggregate_only_and_offline_comparison_separates():
    environment = _env()
    spec, _ = prepare_environment(environment)
    cert = _session(environment)
    quality = run_tap_quality(
        lambda: Agent.model_only(_program()), MockTap, _env(), seed=7,
        build_comparator=MockBlackBox, bootstrap_samples=100,
    )
    assert verify_certificate(cert, spec=spec)
    assert cert["m"] == 0.0 < quality["comparator"]["m"]
    assert set(cert) == {"version", "m_ref", "N", "h_c", "h_s", "t_E",
                         "m", "m_fp", "sig"}


def test_tampered_certificate_fails_verification():
    cert = _session()
    cert["m"] = 1.0 - cert["m"]
    assert not verify_certificate(cert)


@pytest.mark.parametrize(
    "field,value", [("version", "ct/other"), ("m_ref", "wrong"), ("h_s", "wrong")]
)
def test_tampered_protocol_metadata_fails_verification(field, value):
    cert = _session()
    cert[field] = value
    assert not verify_certificate(cert)


def test_malformed_certificate_fails_closed():
    assert not verify_certificate({})


def test_environment_is_fixed_length_and_authenticated_before_parsing(monkeypatch):
    environment = _env()
    spec, plaintext = prepare_environment(environment)
    components = _components()
    assert len(plaintext) == spec["len_E"]
    tampered = bytearray(plaintext)
    tampered[10] ^= 1

    def parsed_before_authentication(*args, **kwargs):
        pytest.fail("environment bytes were parsed before id_E was checked")

    monkeypatch.setattr(certification.json, "loads", parsed_before_authentication)
    with pytest.raises(ValueError, match="digest mismatch"):
        certification.measure(
            components, bytes(tampered), spec,
            (StaticEnvironment,),
        )


def test_environment_source_must_match_registered_workload(monkeypatch):
    environment = _env()
    spec, plaintext = prepare_environment(environment)
    monkeypatch.setattr(certification, "_driver_artifacts", lambda driver: {"changed": "code"})
    with pytest.raises(ValueError, match="not registered"):
        certification.measure(
            _components(), plaintext, spec,
            (StaticEnvironment,),
        )


@pytest.mark.parametrize("certified_first", [False, True])
def test_direct_and_certified_runs_are_comparable(tmp_path, certified_first):
    result = run_comparison(
        ExperimentConfig("mock-comparison", 7, N, "static-treecut"),
        build_agent_package=lambda: agent_package(DATA),
        build_tap_package=tap_package,
        agent_drivers=(MockAgentDriver,), tap_drivers=(MockTapDriver,),
        build_environment=_env, certified_first=certified_first,
    )
    cert = result["certified"]["certificate"]
    assert result["environment_digest"] == prepare_environment(_env())[0]["id_E"]
    assert result["direct"]["measurement"] == {"m": cert["m"], "m_fp": cert["m_fp"]}
    assert result["certified"]["metrics"]["transmitted_bytes_total"] > 0
    assert result["execution_order"] == (
        "certified-first" if certified_first else "direct-first")
    saved = json.loads(write_result(result, tmp_path / "result.json").read_text())
    assert "_reveal" not in saved["certified"]["certificate"]


def test_offline_tap_quality_uses_paired_program_executions():
    result = run_tap_quality(
        lambda: Agent.model_only(_program()), MockTap, _env(), seed=7,
        build_comparator=MockBlackBox, bootstrap_samples=100,
    )
    assert result["tap"]["m"] == 0.0
    assert result["tap"]["roc_auc"] == 1.0
    assert result["tap"]["prediction_rate"] == result["tap"]["mean_score"] == 0.625
    assert result["comparator"]["m"] == result["comparator_minus_tap"]["m"] == 0.625


def test_public_attestation_verifier_never_loads_or_exposes_the_secret(tmp_path, monkeypatch):
    monkeypatch.setattr(attestation, "_KEY_DIR", tmp_path)
    monkeypatch.setattr(attestation, "_SEED_PATH", tmp_path / "manufacturer.key")
    monkeypatch.setattr(attestation, "_PUBLICATION_PATH", tmp_path / "attestation.json")
    initialize_local_attestation()
    publication = attestation._PUBLICATION_PATH.read_bytes()
    signer = StubEnclaveSigner()
    payload, signature = b"attested", signer.sign(b"attested")
    assert attestation._PUBLICATION_PATH.read_bytes() == publication

    monkeypatch.setattr(
        attestation,
        "_create_signing_key",
        lambda: pytest.fail("public verification tried to load the attestation secret"),
    )
    verifier = StubAttestationVerifier.from_publication()
    assert verifier.verify(payload, signature)
    assert verifier.measurement() == signer.measurement()
    assert not hasattr(verifier, "sign")
    assert not any(isinstance(value, SigningKey) for value in vars(verifier).values())


def test_valid_signature_with_unpinned_reference_measurement_is_rejected():
    signing_key = SigningKey.generate()
    signer = StubEnclaveSigner(signing_key)
    verifier = StubAttestationVerifier(bytes(signing_key.verify_key), "pinned-m-ref")
    cert = {
        "version": PROTOCOL_VERSION,
        "m_ref": signer.measurement(),
        "N": "nonce",
        "h_c": "commitment",
        "h_s": "specification",
        "t_E": "environment-message",
        "m": 0.0,
        "m_fp": 0.0,
    }
    cert["sig"] = signer.sign(certification._certificate_payload(cert))

    assert verifier.verify(certification._certificate_payload(cert), cert["sig"])
    assert not certificate_valid(cert, verifier)


def test_separate_process_arguments_reach_the_workload():
    from evals_runner.protocol_integrity_exp import _component_args

    common = {"data": DATA, "monitor_model": None, "detector": None,
              "threshold": 0.25, "device": "cpu", "max_new_tokens": 17}
    confidence = SimpleNamespace(
        **common, case="confidence", program_model="program", dtype="float32",
        model="unused")
    financial = SimpleNamespace(
        **common, case="financial", program_model="unused", dtype="auto",
        model="financial-model")
    assert _component_args(confidence) == [
        "--program-model", "program", "--dtype", "float32",
        "--device", "cpu", "--max-new-tokens", "17",
    ]
    assert _component_args(financial, "probe.npz") == [
        "--model", "financial-model", "--threshold", "0.25",
        "--detector", "probe.npz", "--device", "cpu",
        "--max-new-tokens", "17",
    ]
