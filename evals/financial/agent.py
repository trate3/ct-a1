"""Stateful control program for the insider-trading agent."""

from __future__ import annotations

import inspect

from ct_core.agent import ControlLogic, ToolManifest
from ct_core.interfaces import Generation, ProgramRunner
from ct_core.protocol.digests import sha256_hex

from .actions import SCENARIO_START, TOOL_RESULT, parse_tool_call


class InsiderTradingControl(ControlLogic):
    """Maintain the agent's conversation and act on environment observations.

    The environment supplies an Apollo scenario and later the result of the tool the
    agent actually selected. The control program owns the evolving conversation and model
    invocation. It does not prescribe which tool is selected next.
    """

    def __init__(self) -> None:
        self._history: str | None = None
        self._last_answer: str | None = None

    def digest(self) -> str:
        """Bind the control program and its financial action contract."""
        return sha256_hex(
            super().digest().encode(),
            inspect.getsource(parse_tool_call).encode(),
            SCENARIO_START.encode(),
            TOOL_RESULT.encode(),
        )

    def step(self, model: ProgramRunner, tools: ToolManifest, observation: str,
             capture_layers: list[int] | None = None) -> Generation:
        if observation.startswith(SCENARIO_START):
            self._history = observation.removeprefix(SCENARIO_START).strip()
            self._last_answer = None
        elif observation.startswith(TOOL_RESULT):
            if self._history is None or self._last_answer is None:
                raise ValueError("tool result received before a scenario action")
            result = observation.removeprefix(TOOL_RESULT).strip()
            self._history += f"\n\nASSISTANT: {self._last_answer.strip()}\n\nUSER: {result}"
        else:
            raise ValueError("unknown financial environment observation")

        generation = model.generate(self._history, capture_layers)
        call = parse_tool_call(generation.answer)
        if call.name is not None and not tools.allows(call.name):
            generation = Generation(
                thought=generation.thought,
                answer=f"Undeclared tool call rejected: {call.name}",
                activations=generation.activations,
            )
        self._last_answer = generation.answer
        return generation
