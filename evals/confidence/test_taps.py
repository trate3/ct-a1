"""Wiring tests for the confidence program + tap (scripted FakeBackend, no weights)."""

from __future__ import annotations

import pytest

from ct_core.interfaces import MonitorInput
from evals.confidence.program import GemmaProgram
from evals.confidence.taps import GemmaMonitor
from util.model_backend import FakeBackend, GenOutput


def test_gemma_program_splits_thought_and_answer():
    backend = FakeBackend(on_generate=lambda x: GenOutput(thought="hmm " + x, answer="42"))
    prog = GemmaProgram(backend)
    gen = prog.generate("what is 6*7?")
    assert gen.answer == "42"
    assert gen.thought.startswith("hmm")
    assert "gemma-program-v1" in prog.digest()


def test_tap_sees_thought_comparator_does_not():
    # score encodes whether the prompt included the program's reasoning.
    backend = FakeBackend(on_score=lambda t: 0.9 if "Model reasoning:" in t else 0.1)
    tap = GemmaMonitor(backend, use_thought=True)
    comparator = GemmaMonitor(backend, use_thought=False)
    inp = MonitorInput(x="q", answer="a", thought="the hidden doubt")

    assert tap.predict(inp).z_hat == 1          # tap includes reasoning -> high score
    assert comparator.predict(inp).z_hat == 0   # comparator omits it -> low score
    assert tap.predict(inp).score == pytest.approx(0.9)
    assert "thr=" in tap.digest()               # evaluation identity includes threshold
