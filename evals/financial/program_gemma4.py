"""The probed program for the Gemma-4 financial slice.

`Gemma2Program` loads a dense causal LM with `AutoModelForCausalLM`. Gemma 4 is a
multimodal `Gemma4ForConditionalGeneration`, so three things change and each one is a
place a probe can silently read the wrong thing:

1. **Load class.** `AutoModelForImageTextToText` + `AutoProcessor`, matching
   `util/model_backend/TransformersBackend`. `AutoModelForCausalLM` does not load it.

2. **Where the text layers live.** The text transformer is nested at
   `model.model.language_model.layers`, not `model.model.layers` -- the same nesting
   Apollo's Gemma-4 port has to reach through (`third_party/patches/gemma4-port.patch`,
   the `cut_at_layer` branch). We do not index the stack directly: we read
   `output_hidden_states=True` off the top-level model, which is exactly what Apollo's
   `Activations.from_model` does, so training-time and evaluation-time activations come
   from the same place by construction rather than by coincidence.

3. **Layer coordinates.** `capture_layers` indexes `hidden_states`, whose entry 0 is the
   embedding output and entry i the stream after block i-1. That is the same coordinate
   system as Apollo's `detect_layers`, so a probe's `layers` field transfers unchanged --
   but the layer COUNT differs per checkpoint (26 for gemma-2-2b, 42 for gemma-2-9b and
   for gemma-4-E4B, 60 for gemma-4-31B), so a layer index is only meaningful together
   with the checkpoint it was fitted on. `validate_layers` refuses out-of-range indices
   instead of letting torch raise somewhere less legible.

**Sliding attention.** Gemma 4 alternates sliding-window and full-attention blocks: E4B
has full attention only at blocks 5, 11, 17, 23, 29, 35, 41 of 42. A probe tapped at a
sliding layer reads a stream whose most recent update saw only a 512-token window, which
for a long transcript can exclude the very thing that makes the response deceptive.
Nothing crashes; the failure mode is a plausible-looking score that does not mean what
it appears to. `generate` therefore warns once when asked for a sliding layer. It warns
rather than refuses because a sliding-layer sweep point is a legitimate comparison to
run deliberately -- see probe-track.md.

**Known reconciliation gap, inherited not introduced.** We capture one hidden-state view
per *emitted* token during generation; Apollo trains over the full rendered dialogue in a
single forward pass. probe-track.md flags that difference for the Gemma-2 track already,
and this class reproduces the existing behaviour rather than quietly changing it.
"""

from __future__ import annotations

import time

from ct_core.interfaces import Generation, ProgramRunner


def full_attention_layers(config) -> list[int]:
    """Indices (in `hidden_states` coordinates) whose block uses full attention.

    `layer_types[i]` describes block i; the stream after block i is `hidden_states[i+1]`.
    """
    text = getattr(config, "text_config", config)
    kinds = getattr(text, "layer_types", None)
    if not kinds:
        return []
    return [i + 1 for i, kind in enumerate(kinds) if "sliding" not in str(kind).lower()]


def text_layer_count(config) -> int:
    text = getattr(config, "text_config", config)
    return int(getattr(text, "num_hidden_layers"))


def validate_layers(layers, config) -> None:
    """Fail loudly on layer indices the checkpoint cannot supply."""
    depth = text_layer_count(config)
    bad = [layer for layer in layers if not 0 <= layer <= depth]
    if bad:
        raise ValueError(
            f"probe layers {bad} are outside this checkpoint: hidden_states has "
            f"{depth + 1} entries (0 = embeddings, {depth} = final block). A probe's "
            "layer index is only meaningful for the checkpoint it was fitted on.")


class Gemma4Program(ProgramRunner):
    def __init__(self, model_id: str = "google/gemma-4-E4B-it", device: str = "cuda",
                 dtype: str = "bfloat16", max_new_tokens: int = 128,
                 verbose: bool = False, attn_implementation: str = "eager",
                 enable_thinking: bool = False):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.model_id = model_id
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._verbose = verbose
        self._enable_thinking = enable_thinking
        self._attn_implementation = attn_implementation
        self._warned_sliding = False

        td = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=td, attn_implementation=attn_implementation).to(device)
        self._model.eval()
        self._config = self._model.config
        self._revision = getattr(self._config, "_commit_hash", None) or "unpinned"
        self._full_attention = set(full_attention_layers(self._config))

    def _apply(self, x: str):
        return self._processor.apply_chat_template(
            [{"role": "user", "content": x}], tokenize=True, return_dict=True,
            return_tensors="pt", add_generation_prompt=True,
            enable_thinking=self._enable_thinking,
        ).to(self._model.device)

    def _split(self, text: str, prefix: str) -> tuple[str, str]:
        """Separate the thinking channel from the answer, as the Gemma-4 backend does."""
        try:
            parsed = self._processor.parse_response(text, prefix=prefix)
            return (parsed.get("thinking", "") or "",
                    (parsed.get("content", "") or "").strip())
        except Exception:
            if "<channel|>" in text:
                head, answer = text.split("<channel|>", 1)
                marker = "<|channel>thought"
                thought = head.split(marker, 1)[-1] if marker in head else ""
            else:
                thought, answer = "", text
            for token in ("<turn|>", "<eos>"):
                answer = answer.split(token, 1)[0]
            return thought.strip(), answer.strip()

    def generate(self, x: str, capture_layers: list[int] | None = None) -> Generation:
        torch = self._torch
        if capture_layers:
            validate_layers(capture_layers, self._config)
            sliding = [l for l in capture_layers
                       if self._full_attention and l not in self._full_attention]
            if sliding and not self._warned_sliding:
                self._warned_sliding = True
                print(f"    WARNING: probe layer(s) {sliding} sit after a sliding-window "
                      f"block; full-attention layers are {sorted(self._full_attention)}. "
                      "The probe will read a stream that saw only a local window.",
                      flush=True)

        inputs = self._apply(x)
        prompt_len = inputs["input_ids"].shape[1]
        started = time.time()
        with torch.no_grad():
            generated = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens, do_sample=False,
                return_dict_in_generate=bool(capture_layers),
                output_hidden_states=bool(capture_layers),
            )
        sequences = generated.sequences if capture_layers else generated
        full = sequences[0]
        text = self._processor.decode(full[prompt_len:], skip_special_tokens=False)
        prefix = self._processor.decode(full[:prompt_len], skip_special_tokens=False)
        thought, answer = self._split(text, prefix)

        activations = None
        if capture_layers:
            # One view per emitted token, taken from the decision state that produced
            # that token -- the same contract Gemma2Program provides, so the tap sees
            # the same shape regardless of checkpoint family.
            steps = generated.hidden_states
            if not steps or len(steps) != full.shape[0] - prompt_len:
                raise ValueError("generation did not expose one tap view per emitted token")
            stacked = torch.stack([
                torch.stack([states[layer][0, -1] for layer in capture_layers], dim=0)
                for states in steps
            ], dim=0)  # [generated token, layer, embedding]
            activations = stacked.float().cpu().numpy()

        if self._verbose:
            shape = None if activations is None else activations.shape
            print(f"    program: {time.time() - started:.1f}s | acts {shape} | "
                  f"answer: {answer[:50]!r}", flush=True)
        return Generation(thought=thought, answer=answer, activations=activations)

    def digest(self) -> str:
        return (f"gemma4-program-v1:transformers:{self.model_id}@{self._revision}"
                f":attn={self._attn_implementation}:thinking={self._enable_thinking}")
