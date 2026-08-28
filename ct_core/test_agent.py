"""An Agent commits three components (model + control-logic + tools), not just the model.

GPU-free: a fake ProgramRunner stands in for the model, so these run in CI.
"""

from __future__ import annotations

import pathlib

import pytest

from ct_core.agent import Agent, ControlLogic, ModelOnlyControl, ToolManifest
from ct_core.certification import measure, prepare_components, prepare_environment
from ct_core.interfaces import (
    AgentDriver, ArtifactPackage, EnvironmentTrial, Generation, Monitor, MonitorInput,
    Prediction, ProgramRunner, TapDriver, TestEnvironment,
)
from ct_core.protocol.digests import artifact_commitment_digest


class _FakeModel(ProgramRunner):
    def __init__(self, digest: str = "model-v1"):
        self._digest = digest
        self.seen: tuple[str, list[int] | None] | None = None

    def generate(self, x: str, capture_layers: list[int] | None = None) -> Generation:
        self.seen = (x, capture_layers)
        return Generation(thought="t", answer=f"ans:{x}")

    def digest(self) -> str:
        return self._digest


class _TrialControl(ControlLogic):
    def __init__(self):
        self.index = 0

    def step(self, model, tools, observation, capture_layers=None):
        self.index += 1
        return Generation(str(self.index), observation)


class _TrialTap(Monitor):
    def predict(self, inp):
        value = int(inp.thought == "2")
        return Prediction(value, float(value))

    def digest(self):
        return "tap"


class _TrialAgentDriver(AgentDriver):
    DRIVER = "test-agent/v1"

    def __init__(self, model):
        self.model = model
        self.sessions = []

    @classmethod
    def from_package(cls, config, artifacts):
        if set(config) != {"model"} or set(artifacts) != {"weights"}:
            raise ValueError("bad agent package")
        return cls(config["model"])

    def new_agent(self, randomness):
        control = _TrialControl()
        self.sessions.append(control)
        return Agent(_FakeModel(self.model), control, ToolManifest())


class _TrialTapDriver(TapDriver):
    DRIVER = "test-tap/v1"

    def __init__(self):
        self.sessions = []

    @classmethod
    def from_package(cls, config, artifacts):
        if config or set(artifacts) != {"weights"}:
            raise ValueError("bad tap package")
        return cls()

    def new_tap(self, randomness):
        tap = _TrialTap()
        self.sessions.append(tap)
        return tap


class _TwoStepTrial(EnvironmentTrial):
    def __init__(self, index):
        self.index = index

    def next_query(self, transcript):
        if len(transcript) == 2:
            return None
        return str((self.index - 1) * 2 + len(transcript) + 1)

    def score(self, transcript):
        return 1


class _TwoTrialEnvironment(TestEnvironment):
    DRIVER = "test-two-trial/v1"

    def package_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        if config:
            raise ValueError("unexpected config")
        return cls()

    def trial_count(self):
        return 2

    def max_steps(self):
        return 2

    def new_trial(self, index, randomness):
        return _TwoStepTrial(index)


def test_agent_exposes_the_three_component_digests():
    agent = Agent.model_only(_FakeModel("model-v1"))
    assert agent.model_digest() == "model-v1"
    assert agent.control_digest() == ModelOnlyControl().digest()
    assert agent.tools_digest() == ToolManifest().digest()


def test_model_only_agent_delegates_one_generation():
    model = _FakeModel()
    gen = Agent.model_only(model).generate("hello", capture_layers=[22])
    assert gen.answer == "ans:hello"
    assert model.seen == ("hello", [22])


def test_agent_components_are_explicit():
    with pytest.raises(TypeError):
        Agent(_FakeModel())  # type: ignore[call-arg]


def test_commitment_hashes_actual_package_files_and_config(tmp_path):
    model, tap = tmp_path / "model", tmp_path / "tap"
    model.write_bytes(b"weights-v1")
    tap.write_bytes(b"tap-v1")

    def h_c(model_name="m1"):
        components = prepare_components(
            ArtifactPackage(_TrialAgentDriver.DRIVER, {"model": model_name},
                            (("weights", str(model)),)),
            ArtifactPackage(_TrialTapDriver.DRIVER, {}, (("weights", str(tap)),)),
            (_TrialAgentDriver,),
            (_TrialTapDriver,),
        )
        return artifact_commitment_digest(components.encoded, b"r" * 32)

    base = h_c()
    assert h_c("m2") != base
    model.write_bytes(b"weights-v2")
    assert h_c() != base


def test_manifest_digest_is_version_bound():
    a = ToolManifest((("search", "1.0", "d1"),))
    b = ToolManifest((("search", "1.1", "d1"),))          # version differs
    assert a.digest() != b.digest()
    assert ToolManifest().digest() == ToolManifest(()).digest()


def test_multistep_trials_reset_state_and_aggregate_once_per_trial():
    environment = _TwoTrialEnvironment()
    spec, packed = prepare_environment(environment)
    model, tap = pathlib.Path(__file__), pathlib.Path(__file__)
    components = prepare_components(
        ArtifactPackage(_TrialAgentDriver.DRIVER, {"model": "model-v1"},
                        (("weights", str(model)),)),
        ArtifactPackage(_TrialTapDriver.DRIVER, {}, (("weights", str(tap)),)),
        (_TrialAgentDriver,),
        (_TrialTapDriver,),
    )
    result = measure(components, packed, spec, (_TwoTrialEnvironment,))
    assert result["_trial_predictions"] == result["_trial_labels"] == [1, 1]
    assert result["m"] == result["m_fp"] == 0.0
    assert len(components.agent_driver.sessions) == 2
    assert len(components.tap_driver.sessions) == 2
    assert len({id(session) for session in components.agent_driver.sessions}) == 2
    assert len({id(session) for session in components.tap_driver.sessions}) == 2
