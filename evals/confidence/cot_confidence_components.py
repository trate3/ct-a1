"""Construct the program, monitors, and environment for the CoT-confidence experiment."""

from __future__ import annotations

from functools import lru_cache

from ct_core.agent import Agent
from ct_core.interfaces import AgentDriver, ArtifactPackage, TapDriver
from util.model_backend import (
    TransformersBackend, model_artifact_root, model_artifacts,
)
from util.model_backend import model_backend

from . import program, taps
from .environments import TreeCutEnvironment
from .program import GemmaProgram
from .taps import GemmaMonitor


@lru_cache(maxsize=None)
def _backend(path, device, dtype):
    return TransformersBackend(path, device_map=device, dtype=dtype)


class ConfidenceAgentDriver(AgentDriver):
    DRIVER = "confidence-agent/v1"

    @classmethod
    def code_paths(cls):
        return (program.__file__, model_backend.__file__)

    @classmethod
    def from_package(cls, config, artifacts):
        required = {"program_model", "device", "dtype", "max_new_tokens", "verbose"}
        if set(config) != required:
            raise ValueError("malformed confidence agent package")
        backend = _backend(str(model_artifact_root(artifacts)),
                           config["device"], config["dtype"])
        return cls(backend, config["max_new_tokens"], config["verbose"])

    def __init__(self, backend, max_new_tokens, verbose):
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose

    def new_agent(self, randomness):
        return build_agent(self.backend, self.max_new_tokens, self.verbose)


class ConfidenceTapDriver(TapDriver):
    DRIVER = "confidence-tap/v1"

    @classmethod
    def code_paths(cls):
        return (taps.__file__, model_backend.__file__)

    @classmethod
    def from_package(cls, config, artifacts):
        required = {"monitor_model", "device", "dtype", "verbose"}
        if set(config) != required:
            raise ValueError("malformed confidence tap package")
        backend = _backend(str(model_artifact_root(artifacts)),
                           config["device"], config["dtype"])
        return cls(backend, config["verbose"])

    def __init__(self, backend, verbose):
        self.backend, self.verbose = backend, verbose

    def new_tap(self, randomness):
        return build_tap(self.backend, self.verbose)


def component_packages(program_model, monitor_model=None, device="auto", dtype="auto",
                       max_new_tokens=512, verbose=False):
    _, program_files = model_artifacts(program_model)
    resolved_monitor = monitor_model or program_model
    monitor_files = (program_files if resolved_monitor == program_model
                     else model_artifacts(resolved_monitor)[1])
    agent = ArtifactPackage(
        ConfidenceAgentDriver.DRIVER,
        {"program_model": program_model, "device": device, "dtype": dtype,
         "max_new_tokens": max_new_tokens, "verbose": verbose},
        program_files,
    )
    tap = ArtifactPackage(
        ConfidenceTapDriver.DRIVER,
        {"monitor_model": resolved_monitor, "device": device, "dtype": dtype,
         "verbose": verbose},
        monitor_files,
    )
    return agent, tap


def build_backends(
    program_model: str, monitor_model: str | None, device: str, dtype: str
) -> tuple[TransformersBackend, TransformersBackend]:
    program_path = model_artifacts(program_model)[0]
    program_backend = _backend(str(program_path), device, dtype)
    resolved_monitor = monitor_model or program_model
    monitor_backend = (
        program_backend
        if resolved_monitor == program_model
        else _backend(str(model_artifacts(resolved_monitor)[0]), device, dtype)
    )
    return program_backend, monitor_backend


def build_agent(
    backend: TransformersBackend, max_new_tokens: int = 512, verbose: bool = False
) -> Agent:
    return Agent.model_only(
        GemmaProgram(backend, max_new_tokens=max_new_tokens, verbose=verbose)
    )


def build_tap(backend: TransformersBackend, verbose: bool = False) -> GemmaMonitor:
    return GemmaMonitor(backend, use_thought=True, verbose=verbose)


def build_comparator(backend: TransformersBackend, verbose: bool = False) -> GemmaMonitor:
    return GemmaMonitor(backend, use_thought=False, verbose=verbose)


def build_environment(n: int = 8, seed: int = 7) -> TreeCutEnvironment:
    return TreeCutEnvironment(n, seed)
