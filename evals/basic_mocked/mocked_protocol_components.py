"""Construct agent, tap, comparator, and environment for mock protocol experiments."""

from __future__ import annotations

import json
from pathlib import Path

from ct_core.agent import Agent
from ct_core.interfaces import AgentDriver, ArtifactPackage, TapDriver
from util.dynamic_test_environment import StaticEnvironment

from . import FIXTURE
from . import mocks
from .mocks import GamedTap, MockBlackBox, MockProgram, MockTap


class MockAgentDriver(AgentDriver):
    """Registered mock loader: inert items in, fresh model-only agent per trial."""

    DRIVER = "mock-agent/v1"

    @classmethod
    def code_paths(cls):
        return (mocks.__file__,)

    @classmethod
    def from_package(cls, config, artifacts):
        if set(config) != {"items"} or artifacts or not isinstance(config["items"], list):
            raise ValueError("malformed mock agent package")
        return cls(config["items"])

    def __init__(self, items):
        self.items = items

    def new_agent(self, randomness):
        return Agent.model_only(MockProgram(self.items))


class MockTapDriver(TapDriver):
    """Registered mock tap loader: honest or benchmark-memorizing configuration."""

    DRIVER = "mock-tap/v1"

    @classmethod
    def code_paths(cls):
        return (mocks.__file__,)

    @classmethod
    def from_package(cls, config, artifacts):
        if (set(config) != {"tap", "memorized"} or artifacts
                or config["tap"] not in ("honest", "gamed")
                or not isinstance(config["memorized"], list)
                or any(not isinstance(item, str) for item in config["memorized"])):
            raise ValueError("malformed mock tap package")
        return cls(config["tap"], set(config["memorized"]))

    def __init__(self, tap, memorized):
        self.tap, self.memorized = tap, memorized

    def new_tap(self, randomness):
        return MockTap() if self.tap == "honest" else GamedTap(self.memorized)


def agent_package(data: str | Path = FIXTURE) -> ArtifactPackage:
    items = [json.loads(line) for line in Path(data).read_text().splitlines() if line.strip()]
    return items_package(items)


def items_package(items: list[dict]) -> ArtifactPackage:
    return ArtifactPackage(MockAgentDriver.DRIVER, {"items": items})


def tap_package(tap="honest", memorized=()) -> ArtifactPackage:
    config = {"tap": tap, "memorized": sorted(memorized)}
    return ArtifactPackage(MockTapDriver.DRIVER, config)


def build_agent(data: str | Path = FIXTURE) -> Agent:
    return Agent.model_only(MockProgram.from_jsonl(data))


def build_tap() -> MockTap:
    return MockTap()


def build_comparator() -> MockBlackBox:
    return MockBlackBox()


def build_environment(
    data: str | Path = FIXTURE, n: int = 8, seed: int = 7
) -> StaticEnvironment:
    return StaticEnvironment.from_jsonl(data, n, seed)
