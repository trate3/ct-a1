"""LlamaCppBackend against a scripted transport (no network, no server)."""

from __future__ import annotations

import json
import math

import pytest

requests = pytest.importorskip("requests", reason="needs the [server] extra")

from util.model_backend.llamacpp import LlamaCppBackend  # noqa: E402


class _Response:
    def __init__(self, body, status_code=200):
        self._body, self.status_code = body, status_code
        self.text = json.dumps(body) if isinstance(body, dict) else str(body)

    def json(self):
        return self._body


PROPS = {"model_path": "models/gemma-4-26B-A4B-it.gguf",
         "build_info": "b10173-e9fa0781f"}


@pytest.fixture(autouse=True)
def _stub_props(monkeypatch):
    """/props is fetched with a real GET, so stub it for every test."""
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response(PROPS))


def _transport(chat_body, status_code=200):
    """Records every payload sent, and replays a canned response."""
    sent = []

    def post(url, payload):
        sent.append((url, payload))
        return _Response(chat_body, status_code)

    post.sent = sent
    return post


def _chat(content, reasoning="", finish_reason="stop"):
    return {"choices": [{"finish_reason": finish_reason,
                         "message": {"content": content,
                                     "reasoning_content": reasoning}}]}


def test_splits_reasoning_from_content():
    post = _transport(_chat("The $7 hat is the cheapest.", "Hat 1: $12\nHat 2: $7"))
    backend = LlamaCppBackend("http://localhost:8080", transport=post)
    out = backend.generate([{"role": "user", "content": "which is cheapest?"}],
                           enable_thinking=True, max_new_tokens=600)
    assert out.answer == "The $7 hat is the cheapest."
    assert out.thought.startswith("Hat 1: $12")


def test_thinking_toggle_and_sampling_are_sent():
    post = _transport(_chat("ok"))
    backend = LlamaCppBackend("http://localhost:8080", temperature=1.0, seed=7,
                              transport=post)
    backend.generate([{"role": "user", "content": "hi"}], enable_thinking=False,
                     max_new_tokens=64)
    _, payload = post.sent[-1]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["temperature"] == 1.0 and payload["seed"] == 7
    assert payload["max_tokens"] == 64


def test_server_failure_raises_rather_than_scoring_as_safe():
    post = _transport({"error": "context overflow"}, status_code=500)
    backend = LlamaCppBackend("http://localhost:8080", transport=post)
    with pytest.raises(RuntimeError, match="500"):
        backend.generate([{"role": "user", "content": "hi"}], enable_thinking=True)


def test_body_level_error_also_raises():
    post = _transport({"error": {"message": "bad request"}})
    backend = LlamaCppBackend("http://localhost:8080", transport=post)
    with pytest.raises(RuntimeError, match="bad request"):
        backend.generate([{"role": "user", "content": "hi"}], enable_thinking=True)


def test_empty_content_with_reasoning_is_refused_not_guessed():
    # A mis-split leaves everything in reasoning_content.
    post = _transport(_chat("", "the whole answer ended up in here"))
    backend = LlamaCppBackend("http://localhost:8080", transport=post)
    with pytest.raises(RuntimeError, match="reasoning-format"):
        backend.generate([{"role": "user", "content": "hi"}], enable_thinking=True)


def test_truncated_generation_returns_empty_answer_instead_of_raising():
    # Hitting the token cap mid-thought is truncation, not a mis-split.
    post = _transport(_chat("", "still thinking when the budget ran out",
                            finish_reason="length"))
    backend = LlamaCppBackend("http://localhost:8080", transport=post)
    out = backend.generate([{"role": "user", "content": "hi"}], enable_thinking=True)
    assert out.answer == "" and out.thought


def test_score_yes_no_sums_token_mass():
    body = {"choices": [{"message": {"content": "yes"}, "logprobs": {"content": [
        {"top_logprobs": [{"token": "yes", "logprob": math.log(0.75)},
                          {"token": "Yes", "logprob": math.log(0.05)},
                          {"token": "no", "logprob": math.log(0.20)}]}]}}]}
    backend = LlamaCppBackend("http://localhost:8080", transport=_transport(body))
    assert backend.score_yes_no([{"role": "user", "content": "q"}]) == pytest.approx(0.8)


def test_score_yes_no_without_logprobs_raises():
    backend = LlamaCppBackend("http://localhost:8080", transport=_transport(_chat("yes")))
    with pytest.raises(RuntimeError, match="logprobs"):
        backend.score_yes_no([{"role": "user", "content": "q"}])


def test_digest_binds_server_identity_and_sampling():
    post = _transport(_chat("ok"))
    plain = LlamaCppBackend("http://localhost:8080", transport=post)
    assert plain.digest() == "llamacpp:models/gemma-4-26B-A4B-it.gguf@b10173-e9fa0781f"

    sampled = LlamaCppBackend("http://localhost:8080", temperature=1.0, seed=7,
                              transport=post)
    assert sampled.digest().endswith(":t=1.0:seed=7")
    # A hash of the actual weights supersedes the server's self-report.
    hashed = LlamaCppBackend("http://localhost:8080", model_digest="sha256:abc",
                             transport=post)
    assert hashed.digest() == "llamacpp:sha256:abc"


def test_onion_host_routes_over_socks5h():
    post = _transport(_chat("ok"))
    backend = LlamaCppBackend("http://abc.onion", transport=post)
    assert backend._proxies["http"].startswith("socks5h://")
    assert LlamaCppBackend("http://localhost:8080", transport=post)._proxies is None


def test_socks_proxy_is_overridable(monkeypatch):
    # Selects a Tor instance other than the local default.
    post = _transport(_chat("ok"))
    monkeypatch.setenv("CT_SOCKS_PROXY", "socks5h://localhost:9150")
    backend = LlamaCppBackend("http://abc.onion", transport=post)
    assert backend._proxies["http"] == "socks5h://localhost:9150"


def test_dropped_connection_is_retried_not_fatal():
    import requests as _rq

    calls = []

    def flaky(url, payload):
        calls.append(url)
        if len(calls) <= 2:
            raise _rq.exceptions.ConnectionError("Remote end closed connection")
        return _Response(_chat("recovered"))

    backend = LlamaCppBackend("http://localhost:8080", retries=4, backoff=0,
                              transport=flaky)
    out = backend.generate([{"role": "user", "content": "hi"}], enable_thinking=True)
    assert out.answer == "recovered"


def test_retries_are_bounded_and_then_raise():
    import requests as _rq

    def always_down(url, payload):
        raise _rq.exceptions.ConnectionError("down")

    backend = LlamaCppBackend("http://localhost:8080", retries=2, backoff=0,
                              transport=always_down)
    with pytest.raises(RuntimeError, match="unreachable after 3 attempts"):
        backend.generate([{"role": "user", "content": "hi"}], enable_thinking=True)


def test_http_errors_are_not_retried():
    # A 500 is the server telling us something; only transport faults are transient.
    calls = []

    def failing(url, payload):
        calls.append(url)
        return _Response({"error": "boom"}, status_code=500)

    backend = LlamaCppBackend("http://localhost:8080", retries=4, backoff=0,
                              transport=failing)
    with pytest.raises(RuntimeError, match="500"):
        backend.generate([{"role": "user", "content": "hi"}], enable_thinking=True)
    assert len(calls) == 1  # the chat only -- no retry storm


def test_identity_uses_get_and_warns_when_it_cannot_be_read(monkeypatch, capsys):
    # /props answers GET only; binding "unknown" must be loud, not silent.
    seen = {}
    monkeypatch.setattr(requests, "get",
                        lambda url, **k: seen.setdefault("url", url) and None
                        or _Response(PROPS))
    backend = LlamaCppBackend("http://localhost:8080", transport=_transport(_chat("ok")))
    assert seen["url"].endswith("/props")
    assert "gemma-4-26B-A4B" in backend.digest()

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response({}, status_code=405))
    degraded = LlamaCppBackend("http://localhost:8080", transport=_transport(_chat("ok")))
    assert degraded.digest() == "llamacpp:unknown@unknown"
    assert "WARNING" in capsys.readouterr().out


def test_weights_picks_the_language_model_not_the_projector(tmp_path, monkeypatch):
    from util.model_backend import llamacpp

    (tmp_path / "gemma-4-26B_q4_0-it.gguf").write_bytes(b"x" * 5000)
    (tmp_path / "gemma-4-26B-mmproj.gguf").write_bytes(b"x" * 9000)   # bigger, but wrong
    monkeypatch.setattr(llamacpp, "snapshot_download", lambda *a, **k: str(tmp_path),
                        raising=False)
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub",
                        type("m", (), {"snapshot_download": lambda *a, **k: str(tmp_path)}))
    assert llamacpp._weights("some/repo").endswith("gemma-4-26B_q4_0-it.gguf")


def test_missing_binary_says_what_to_do(monkeypatch):
    from util.model_backend import llamacpp

    monkeypatch.delenv("LLAMA_SERVER_BIN", raising=False)
    monkeypatch.setattr(llamacpp.shutil, "which", lambda name: None)
    monkeypatch.setattr(llamacpp.os.path, "exists", lambda p: False)
    with pytest.raises(RuntimeError, match="LLAMA_SERVER_BIN"):
        llamacpp._binary()
