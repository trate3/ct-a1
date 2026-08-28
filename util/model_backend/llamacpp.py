"""InferenceBackend over a llama.cpp OpenAI-compatible server.

Serves models `TransformersBackend` cannot: bitsandbytes only replaces `nn.Linear`, so
a MoE checkpoint with fused expert tensors stays unquantized.

A remote server runs the model outside the trusted workload, and `digest()` then binds
only what the server reports about itself; `serve()` runs one on loopback, and
`model_digest` binds weights hashed elsewhere.

Requires the [server] extra; `requests` is imported lazily.
"""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import socket
import subprocess
import time
from urllib.parse import urlparse

from .model_backend import GenOutput, InferenceBackend


class LlamaCppBackend(InferenceBackend):
    _YES = ("yes", "yeah", "true")
    _NO = ("no", "nope", "false")

    def __init__(self, base_url: str, *, temperature: float | None = None,
                 seed: int | None = None, timeout: int = 900,
                 model_digest: str | None = None, retries: int = 4,
                 backoff: float = 3.0, transport=None):
        import requests

        self.base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._seed = seed
        self._timeout = timeout
        self._model_digest = model_digest
        self._retries = retries
        self._backoff = backoff
        self._proxies = None

        host = urlparse(self.base_url).hostname or ""
        if host.endswith(".onion"):
            # socks5h resolves at the proxy: a .onion has no DNS entry here.
            # CT_SOCKS_PROXY selects a different Tor instance.
            proxy = os.environ.get("CT_SOCKS_PROXY", "socks5h://localhost:9050")
            self._proxies = {"http": proxy, "https": proxy}
        if host not in ("127.0.0.1", "localhost", "::1"):
            print(f"[llamacpp] NOTE: {host} is not loopback. The program's thought "
                  "channel leaves this machine; evaluation only, not a certifiable "
                  "deployment.", flush=True)

        self._post = transport or (
            lambda url, payload: requests.post(url, json=payload, proxies=self._proxies,
                                               timeout=self._timeout)
        )
        self._transient = (requests.exceptions.RequestException, OSError)
        self._model, self._build = self._identify()

    def _send(self, url: str, payload: dict | None):
        """POST, retrying transport faults with exponential backoff.

        An HTTP status or malformed body raises instead: those come from the server.
        """
        last = None
        for attempt in range(self._retries + 1):
            try:
                return self._post(url, payload)
            except self._transient as exc:
                last = exc
                if attempt == self._retries:
                    break
                delay = self._backoff * (2 ** attempt)
                print(f"[llamacpp] {type(exc).__name__} on attempt {attempt + 1}"
                      f"/{self._retries + 1}; retrying in {delay:.0f}s", flush=True)
                time.sleep(delay)
        raise RuntimeError(
            f"llama.cpp unreachable after {self._retries + 1} attempts: "
            f"{type(last).__name__}: {str(last)[:200]}"
        ) from last

    def _chat(self, payload: dict) -> dict:
        response = self._send(f"{self.base_url}/v1/chat/completions", payload)
        if response.status_code != 200:
            # Abort the round rather than resolve a failure to a verdict.
            raise RuntimeError(
                f"llama.cpp server returned {response.status_code}: {response.text[:400]}"
            )
        body = response.json()
        if "error" in body:
            raise RuntimeError(f"llama.cpp server error: {str(body['error'])[:400]}")
        return body

    def _identify(self) -> tuple[str, str]:
        """Model path and build for the digest: the server's self-report, not a hash."""
        import requests

        try:
            response = requests.get(f"{self.base_url}/props", proxies=self._proxies,
                                    timeout=self._timeout)
            props = response.json() if response.status_code == 200 else {}
        except Exception:
            props = {}
        model = props.get("model_path")
        build = props.get("build_info")
        if not model or not build:
            print("[llamacpp] WARNING: could not read /props, so the certificate will "
                  "bind 'unknown' instead of the served model. Pass model_digest to "
                  "bind the weights explicitly.", flush=True)
        return model or "unknown", build or "unknown"

    def _sampling(self) -> dict:
        params = {}
        if self._temperature is not None:
            params["temperature"] = self._temperature
        if self._seed is not None:
            params["seed"] = self._seed
        return params

    def generate(self, messages: list[dict], *, enable_thinking: bool,
                 max_new_tokens: int = 512) -> GenOutput:
        body = self._chat({
            "messages": messages,
            "max_tokens": max_new_tokens,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            **self._sampling(),
        })
        choice = body["choices"][0]
        message = choice["message"]
        answer = (message.get("content") or "").strip()
        thought = (message.get("reasoning_content") or "").strip()

        if not answer and thought and choice.get("finish_reason") != "length":
            # Nothing outside the thought channel: --reasoning-format is mis-splitting,
            # and guessing which half is the answer corrupts the transcript.
            raise RuntimeError(
                "llama.cpp returned an empty content with a non-empty reasoning_content "
                "and did not stop on length. The server's --reasoning-format is likely "
                "mis-splitting this model's thought channel."
            )
        return GenOutput(thought=thought, answer=answer)

    def score_yes_no(self, messages: list[dict]) -> float:
        """P('yes') / (P('yes') + P('no')) over the first generated token.

        top_logprobs is pre-sampling, so the reading is undistorted by temperature.
        Thinking is off, or the first token starts a thought rather than the verdict.
        """
        body = self._chat({
            "messages": messages,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False},
            **self._sampling(),
        })
        entries = (body["choices"][0].get("logprobs") or {}).get("content") or []
        if not entries:
            raise RuntimeError(
                "llama.cpp returned no logprobs. score_yes_no needs a build with "
                "OpenAI-compatible logprobs (llama.cpp PR #10783 or later)."
            )
        p_yes = p_no = 0.0
        for token in entries[0].get("top_logprobs", []):
            text = token.get("token", "").strip().lower()
            probability = math.exp(token.get("logprob", -math.inf))
            if text in self._YES:
                p_yes += probability
            elif text in self._NO:
                p_no += probability
        total = p_yes + p_no
        return p_yes / total if total > 0 else 0.5

    def digest(self) -> str:
        # Self-reported unless model_digest supplied a hash of the weights.
        identity = self._model_digest or f"{self._model}@{self._build}"
        base = f"llamacpp:{identity}"
        if self._temperature is None:
            return base
        return f"{base}:t={self._temperature}:seed={self._seed}"



# --- running our own server -------------------------------------------------

DEFAULT_GGUF_REPO = "google/gemma-4-26B-A4B-it-qat-q4_0-gguf"


def _binary() -> str:
    """The llama-server executable: $LLAMA_SERVER_BIN, PATH, or the usual build dir."""
    candidates = [os.environ.get("LLAMA_SERVER_BIN"), shutil.which("llama-server"),
                  os.path.expanduser("~/llama.cpp/build/bin/llama-server")]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise RuntimeError(
        "llama-server not found. Build llama.cpp, or set LLAMA_SERVER_BIN to the binary."
    )


def _weights(repo: str) -> str:
    """Download the repo's GGUF weights if not already cached, and return the path."""
    from huggingface_hub import snapshot_download

    directory = snapshot_download(repo, allow_patterns=["*.gguf"])
    # Repos ship a vision projector alongside the weights.
    ggufs = [os.path.join(directory, f) for f in os.listdir(directory)
             if f.endswith(".gguf") and "mmproj" not in f]
    if not ggufs:
        raise RuntimeError(f"no usable .gguf in {repo}")
    return max(ggufs, key=os.path.getsize)


@contextlib.contextmanager
def serve(repo: str = DEFAULT_GGUF_REPO, *, context: int = 16384, load_timeout: int = 900):
    """Fetch the weights if absent, serve them on loopback, and shut down on exit.

    `--cpu-moe` keeps the routed experts in system RAM and the rest on the GPU, so a
    26B MoE serves from a few GB of VRAM.
    """
    import requests

    model = _weights(repo)
    with socket.socket() as probe:                 # a free port
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"

    print(f"[llamacpp] starting a server for {repo} on {url}", flush=True)
    process = subprocess.Popen(
        [_binary(), "-m", model, "--host", "127.0.0.1", "--port", str(port),
         "-ngl", "99", "--cpu-moe", "--no-mmap", "-c", str(context), "--jinja"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + load_timeout
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {process.returncode}")
            try:
                if requests.get(f"{url}/health", timeout=5).status_code == 200:
                    break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError(f"llama-server did not come up within {load_timeout}s")
        print(f"[llamacpp] server ready on {url}", flush=True)
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        print("[llamacpp] server stopped", flush=True)
