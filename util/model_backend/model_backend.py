"""Inference backend: the one place model specifics live.

The program and monitors call an `InferenceBackend`; swapping the backend swaps
mock <-> real without touching protocol or application code.

SOURCES (imported directly from Hugging Face by model id; nothing copied):
  - Gemma 4 (Google DeepMind, Apache-2.0), released 2026-04-02.
    model card:  https://ai.google.dev/gemma/docs/core/model_card_4
    HF (E2B-it): https://huggingface.co/google/gemma-4-E2B-it   (NOT gated as of 2026-07-27)
    transformers docs: https://huggingface.co/docs/transformers/model_doc/gemma4
  - Thinking / channel control tokens and the P(yes) readout convention are
    documented inline on TransformersBackend where they are used.

Requires `transformers>=5.10.1` + torch (imported lazily, only when the real
backend is constructed) -- installable via the project's [model] extra. The mock
harness and tests never import them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class GenOutput:
    thought: str
    answer: str


class InferenceBackend:
    def generate(self, messages: list[dict], *, enable_thinking: bool,
                 max_new_tokens: int = 512) -> GenOutput:
        """Run a chat and return the split (thought, answer)."""
        raise NotImplementedError

    def score_yes_no(self, messages: list[dict]) -> float:
        """P('yes') / (P('yes') + P('no')) on the next token (thinking off)."""
        raise NotImplementedError

    def digest(self) -> str:
        raise NotImplementedError


def model_artifacts(model: str) -> tuple[Path, tuple[tuple[str, str], ...]]:
    """Resolve a model ID or local directory to the exact files provisioned to a driver."""
    source = Path(model)
    if not source.is_dir():
        from huggingface_hub import snapshot_download

        source = Path(snapshot_download(model))
    source = source.resolve()
    artifacts = tuple(
        (f"model/{path.relative_to(source).as_posix()}", str(path))
        for path in sorted(source.rglob("*")) if path.is_file()
    )
    if not artifacts:
        raise ValueError(f"model artifact directory is empty: {source}")
    return source, artifacts


def artifact_total_bytes(model: str) -> int:
    """Total provisioned bytes for a model, the quantity commitment hashing is linear in.

    Committing a component hashes every provisioned file, so this is the input to that
    stage's cost model and belongs in any record of what certification cost.
    """
    return sum(Path(path).stat().st_size for _, path in model_artifacts(model)[1])


def model_artifact_root(artifacts: dict[str, str | bytes]) -> Path:
    """Recover and validate the model directory represented by named artifact paths."""
    entries = [(PurePosixPath(name).relative_to("model"), value)
               for name, value in artifacts.items()]
    if not entries or any(not isinstance(value, str) for _, value in entries):
        raise ValueError("model artifacts must be provisioned files")
    relative, location = entries[0]
    root = Path(location)
    for _ in relative.parts:
        root = root.parent
    if any(Path(value) != root.joinpath(*path.parts) for path, value in entries):
        raise ValueError("model artifact paths do not match their package names")
    return root


class TransformersBackend(InferenceBackend):
    """Gemma-4 via Hugging Face transformers (imported directly by model id).

    Model-specific facts, all from the Gemma 4 model card / tokenizer (verified
    2026-07-27) and isolated here so the rest of the code stays model-agnostic:
      - load with the multimodal auto class (Gemma 4 is Gemma4ForConditionalGeneration
        even for text): AutoModelForImageTextToText + AutoProcessor.
      - thinking is a chat-template kwarg `enable_thinking`; when on, the model emits
        `<|channel>thought ... <channel|>{answer}`. `processor.parse_response(text)`
        returns {"thinking", "content", ...}; we use that, else split on `<channel|>`.
      - yes/no readout: single forward pass, next-token logits, sum prob mass over
        the {yes, Yes, ' yes', ' Yes'} token ids vs {no, ...}. IDs are resolved from
        the live tokenizer at construction (do not hardcode).
    """

    _YES_WORDS = ["yes", "Yes", " yes", " Yes"]
    _NO_WORDS = ["no", "No", " no", " No"]

    def __init__(self, model_id: str = "google/gemma-4-E2B-it", device_map: str = "auto",
                 dtype: str = "auto", revision: str | None = None,
                 temperature: float | None = None, seed: int | None = None):
        # Lazy heavy imports: only when a real backend is actually constructed.
        import torch  # noqa: F401
        from transformers import AutoModelForImageTextToText, AutoProcessor

        # "auto" asks accelerate to place layers by fit, which on a memory-tight
        # machine silently offloads the tail to DISK -- correct but so slow a session
        # times out. Resolve to a single concrete device so the whole model lives in
        # one place. Pass device_map="cpu" if the accelerator runs out of memory, or a
        # real device map for a multi-GPU server.
        if device_map == "auto":
            device_map = ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

        # Sampling is off by default, keeping greedy decoding; `seed` makes a sampled
        # run replayable. Both are bound into digest().
        self.model_id = model_id
        self._temperature = temperature
        self._seed = seed
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, device_map=device_map, revision=revision)
        # Resolve the actual checkpoint commit so the digest binds the real weights,
        # not just the model name (pin --revision to make it reproducible).
        self._revision = revision or getattr(self._model.config, "_commit_hash", None) or "unpinned"
        tok = self._processor.tokenizer
        self._yes_ids = self._single_token_ids(tok, self._YES_WORDS)
        self._no_ids = self._single_token_ids(tok, self._NO_WORDS)

    @staticmethod
    def _single_token_ids(tok, words: list[str]) -> list[int]:
        ids = []
        for w in words:
            enc = tok(w, add_special_tokens=False).input_ids
            if len(enc) == 1:
                ids.append(enc[0])
        if not ids:
            raise RuntimeError("no single-token yes/no variants found for this tokenizer")
        return sorted(set(ids))

    def _apply(self, messages, enable_thinking, add_generation_prompt=True):
        return self._processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking,
        ).to(self._model.device)

    def generate(self, messages: list[dict], *, enable_thinking: bool,
                 max_new_tokens: int = 512) -> GenOutput:
        inputs = self._apply(messages, enable_thinking)
        if self._seed is not None:
            self._torch.manual_seed(self._seed)
        sampling = ({"do_sample": True, "temperature": self._temperature}
                    if self._temperature is not None else {"do_sample": False})
        out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling)
        gen = out[0, inputs["input_ids"].shape[1]:]
        text = self._processor.decode(gen, skip_special_tokens=False)
        prefix = self._processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
        thought, answer = self._split(text, prefix)
        return GenOutput(thought=thought, answer=answer)

    def _split(self, text: str, prefix: str = "") -> tuple[str, str]:
        # Prefer the documented parser; fall back to the channel delimiters.
        try:
            parsed = self._processor.parse_response(text, prefix=prefix)
            return parsed.get("thinking", "") or "", (parsed.get("content", "") or "").strip()
        except Exception:
            if "<channel|>" in text:
                head, answer = text.split("<channel|>", 1)
                marker = "<|channel>thought"
                thought = head.split(marker, 1)[-1] if marker in head else ""
            else:
                thought, answer = "", text
            for t in ("<turn|>", "<eos>"):
                answer = answer.split(t, 1)[0]
            return thought.strip(), answer.strip()

    def score_yes_no(self, messages: list[dict]) -> float:
        inputs = self._apply(messages, enable_thinking=False)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits[0, -1, :]
        probs = logits.softmax(-1)
        p_yes = probs[self._yes_ids].sum().item()
        p_no = probs[self._no_ids].sum().item()
        denom = p_yes + p_no
        return p_yes / denom if denom > 0 else 0.5

    def residual_activations(self, x: str, layers: list[int]):
        """Residual-stream activations for the probe track (ActivationSource duck type).

        Mirrors Apollo's Activations.from_model: a forward pass with
        output_hidden_states=True, reading hidden_states[layer] (hidden_states[0] is
        the embedding, [i] is the output of block i). Returns numpy [num_tokens,
        len(layers), emb]. NOT exercised on real weights in CI.
        """
        inputs = self._apply([{"role": "user", "content": x}], enable_thinking=False,
                             add_generation_prompt=False)
        with self._torch.no_grad():
            hs = self._model(**inputs, output_hidden_states=True).hidden_states
        stacked = self._torch.stack([hs[i][0] for i in layers], dim=1)  # [tokens, L, emb]
        return stacked.float().cpu().numpy()

    def digest(self) -> str:
        # Binds the model id and resolved revision; sampling only when set, so greedy
        # backends keep their original digest string.
        base = f"transformers:{self.model_id}@{self._revision}"
        if self._temperature is None:
            return base
        return f"{base}:t={self._temperature}:seed={self._seed}"


def user_text(messages: list[dict]) -> str:
    """The last user turn's text (backends key on this)."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            c = msg["content"]
            return c if isinstance(c, str) else " ".join(
                p.get("text", "") for p in c if isinstance(p, dict))
    return ""


class FakeBackend(InferenceBackend):
    """Scripted backend for tests / offline wiring checks (no weights, no deps).

    `on_generate(user_text) -> GenOutput` and `on_score(user_text) -> float`.
    """

    def __init__(self, on_generate=None, on_score=None, name: str = "fake"):
        self._gen = on_generate or (lambda t: GenOutput("", ""))
        self._score = on_score or (lambda t: 0.5)
        self._name = name

    def generate(self, messages, *, enable_thinking: bool, max_new_tokens: int = 512) -> GenOutput:
        return self._gen(user_text(messages))

    def score_yes_no(self, messages) -> float:
        return self._score(user_text(messages))

    def digest(self) -> str:
        return f"fake:{self._name}"
