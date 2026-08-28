"""Tests for the shared model backend (the scripted FakeBackend)."""

from __future__ import annotations

from util.model_backend import FakeBackend, GenOutput, TransformersBackend, user_text


def test_fake_backend_generate_and_score():
    b = FakeBackend(on_generate=lambda t: GenOutput(thought="th", answer="ans:" + t),
                    on_score=lambda t: 0.8)
    out = b.generate([{"role": "user", "content": "hi"}], enable_thinking=True)
    assert out.answer == "ans:hi" and out.thought == "th"
    assert b.score_yes_no([{"role": "user", "content": "q"}]) == 0.8
    assert "fake" in b.digest()


def test_user_text_extracts_last_user_turn():
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert user_text(msgs) == "a"


def test_transformers_response_parser_receives_prompt_prefix():
    class Processor:
        def parse_response(self, text, *, prefix):
            assert text == "generated"
            assert prefix == "prompt"
            return {"thinking": "reasoning", "content": "answer"}

    backend = object.__new__(TransformersBackend)
    backend._processor = Processor()
    assert backend._split("generated", "prompt") == ("reasoning", "answer")
