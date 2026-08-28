"""Agent as an object to commit in the certified probes framework.

A tap certifies an agent, so the commitment must bind the whole agent, not just the model
powering it. An agent is a single logical program that calls a model and orchestrates its
behavior; committing it binds three components:
1. the model (frozen weights)
2. the control-logic program that wraps it
3. the manifest of tools it may call (with versions).
The trusted workload reconstructs them from an inert artifact package and hashes the
actual package bytes, so a certificate does not trust these objects to identify
themselves.

`Agent.model_only` names the narrower case used by evaluations of a model itself. General
agent construction requires all three components explicitly.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass

from .interfaces import Generation, ProgramRunner
from .protocol.digests import sha256_hex


@dataclass(frozen=True)
class ToolManifest:
    """The tools an agent may call, pinned by version.

    Each entry is (tool_id, version, digest). Committing the manifest commits the tool
    versions, so a certificate reads "this agent with tool X at version v" (frozen-tool
    composition). Empty when the agent calls no tools.
    """

    tools: tuple[tuple[str, str, str], ...] = ()

    def digest(self) -> str:
        canonical = json.dumps([list(t) for t in self.tools], sort_keys=True,
                               separators=(",", ":"))
        return sha256_hex(canonical.encode())

    def ids(self) -> tuple[str, ...]:
        return tuple(tool_id for tool_id, _, _ in self.tools)

    def allows(self, tool_id: str) -> bool:
        return tool_id in self.ids()


class ControlLogic:
    """The program that runs the model and coordinates its available tools.

    A control program may keep state across environment rounds. Each call to `step`
    receives the environment's latest observation and returns the agent's actual response.
    """

    def step(self, model: ProgramRunner, tools: ToolManifest, observation: str,
             capture_layers: list[int] | None = None) -> Generation:
        raise NotImplementedError

    def digest(self) -> str:
        """Evaluation-facing identity of this concrete control program."""
        return sha256_hex(inspect.getsource(type(self)).encode())

class ModelOnlyControl(ControlLogic):
    """One model invocation per environment round, for explicitly model-only evaluations."""

    def step(self, model: ProgramRunner, tools: ToolManifest, observation: str,
             capture_layers: list[int] | None = None) -> Generation:
        return model.generate(observation, capture_layers)


class Agent:
    """The program under test: a model wrapped by control logic and a tool manifest.

    Its registered package includes all three, so h_c binds the agent, not just the model.
    The control program owns model invocation and state across environment rounds. A tap
    receives the thought and activations returned by that actual execution.
    """

    def __init__(self, model: ProgramRunner, control: ControlLogic, tools: ToolManifest):
        self._model = model
        self._control = control
        self._tools = tools

    @classmethod
    def model_only(cls, model: ProgramRunner) -> Agent:
        """Construct the deliberate model-only case used by the mock and CoT evaluations."""
        return cls(model, ModelOnlyControl(), ToolManifest())

    def generate(self, x: str, capture_layers: list[int] | None = None) -> Generation:
        return self._control.step(self._model, self._tools, x, capture_layers)

    def model_digest(self) -> str:
        return self._model.digest()

    def control_digest(self) -> str:
        return self._control.digest()

    def tools_digest(self) -> str:
        return self._tools.digest()
