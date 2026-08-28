"""The seams the Certified Taps engine talks to.

The certification path is written purely against these interfaces, so a program,
a monitor (tap), or a test environment can be a mock, a real Gemma model, or an
Apollo probe without the protocol code changing. Concrete implementations live in
`evals/` (purpose-specific) or `util/` (shared).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Generation:
    thought: str            # stays inside the enclave (computation trace)
    answer: str             # response y_i retained inside the enclave during certification
    activations: object = None  # residual activations from this program invocation,
                                # for a white-box probe tap; None unless requested.


class ProgramRunner:
    """The model an agent runs, a component of the committed agent (Pi).

    An agent (`ct_core/agent.py`) wraps a `ProgramRunner` with control logic and a tool
    manifest; the enclave runs the model through the agent and commits all three.
    """

    def generate(self, x: str, capture_layers: list[int] | None = None) -> Generation:
        """Run the program on x.

        `capture_layers`, when given, asks the program to also return residual-stream
        activations at those layers for the same response -- so a probe tap reads the
        committed program's activations rather than running a different model. Text
        taps leave it None.
        """
        raise NotImplementedError

    def digest(self) -> str:
        """Evaluation-facing identity; certification hashes package bytes instead."""
        raise NotImplementedError

@dataclass(frozen=True)
class MonitorInput:
    x: str
    answer: str
    thought: str | None = None    # provided only to the tap; None for the comparator
    activations: object = None    # residual activations from the program's run, for a probe


@dataclass(frozen=True)
class Prediction:
    z_hat: int      # certified binary prediction (1 = unsafe -> block)
    score: float    # continuous confidence, for offline AUC only


class Monitor:
    """A tap (certified) or the black-box comparator (baseline)."""

    # A white-box probe tap sets these so the enclave captures activations from the
    # program's execution and passes them in; text taps leave them as-is.
    needs_activations: bool = False
    probe_layers: list[int] | None = None

    def predict(self, inp: MonitorInput) -> Prediction:
        raise NotImplementedError

    def digest(self) -> str:
        """Evaluation-facing identity; certification hashes package bytes instead."""
        raise NotImplementedError

class EnvironmentTrial:
    """One fresh E instance, exposed only through its transcript interface."""

    def next_query(self, transcript: tuple[tuple[str, str], ...]) -> str | None:
        """Return the next query, or None to halt the trial."""
        raise NotImplementedError

    def score(self, transcript: tuple[tuple[str, str], ...]) -> int:
        """Return the completed trial's binary label."""
        raise NotImplementedError


class TestEnvironment:
    """A certifier-side builder for an attested environment driver and private config.

    The wire carries only the canonical package returned by ``package_config`` plus
    the exact source artifacts named by ``code_paths``. The trusted workload matches
    those bytes to a registered driver before constructing any environment object.
    """

    DRIVER = ""

    def package_config(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict) -> "TestEnvironment":
        raise NotImplementedError

    @classmethod
    def code_paths(cls) -> tuple[str, ...]:
        """Additional source artifacts that determine E; the class module is implicit."""
        return ()

    def runtime_driver(self) -> type["TestEnvironment"]:
        return type(self)

    def trial_count(self) -> int:
        raise NotImplementedError

    def max_steps(self) -> int:
        raise NotImplementedError

    def new_trial(self, index: int, randomness: bytes) -> EnvironmentTrial:
        """Create independent trial ``index`` from enclave-generated coins."""
        raise NotImplementedError


@dataclass(frozen=True)
class ArtifactPackage:
    """Inert provider input naming registered code, config, and artifacts."""

    driver: str
    config: dict
    artifacts: tuple[tuple[str, str | bytes], ...] = ()


class RegisteredDriver:
    """Code installed in, and attested as part of, the trusted workload."""

    DRIVER = ""

    @classmethod
    def code_paths(cls) -> tuple[str, ...]:
        """Additional implementation files; the class module is included automatically."""
        return ()


class AgentDriver(RegisteredDriver):
    """Trusted loader for an agent package and factory for fresh trial agents."""

    @classmethod
    def from_package(cls, config: dict, artifacts: dict[str, str | bytes]) -> "AgentDriver":
        raise NotImplementedError

    def new_agent(self, randomness: bytes) -> "Agent":
        """Construct a fresh agent for one independent trial."""
        raise NotImplementedError


class TapDriver(RegisteredDriver):
    """Trusted loader for a tap package and factory for fresh trial taps."""

    @classmethod
    def from_package(cls, config: dict, artifacts: dict[str, str | bytes]) -> "TapDriver":
        raise NotImplementedError

    def new_tap(self, randomness: bytes) -> Monitor:
        """Construct a fresh tap for one independent trial."""
        raise NotImplementedError
