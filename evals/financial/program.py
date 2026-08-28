"""The probed program for the financial slice: Gemma-2 as a plain causal LM.

Apollo's probe pipeline supports Gemma-2 / Llama / Mistral, NOT Gemma-4, so the probe
is trained on Gemma-2 activations. This program loads that same model with
`AutoModelForCausalLM` and retains the selected layer states from the generation calls
that produced the answer. It is the Gemma-2 counterpart to
`evals/confidence/program.py` (which targets Gemma-4). Not exercised in CI (gated weights).
"""

from __future__ import annotations

import time

from ct_core.interfaces import Generation, ProgramRunner


class Gemma2Program(ProgramRunner):
    def __init__(self, model_id: str = "google/gemma-2-2b-it", device: str = "cpu",
                 dtype: str = "float32", max_new_tokens: int = 8, verbose: bool = False,
                 attn_implementation: str = "eager"):
        # Lazy heavy imports (torch/transformers), only when a real program is built.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self._device = device
        td = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        # float32 on CPU is full-precision and multi-threaded; bf16 on Apple-Silicon CPU
        # is single-threaded and slow, and MPS can't load Gemma-2's bf16 checkpoint.
        # Eager attention, matching the loader Apollo's probe trainer uses. Gemma-2
        # applies attention-logit soft-capping, which the fused SDPA and
        # flash-attention kernels do not implement, so they compute a different
        # attention than eager does. The probe's directions were fitted on eager
        # activations; reading sdpa activations with them would apply the tap off the
        # distribution it was trained on, and nothing would raise. Pinned rather than
        # left to the default because that default varies by transformers version and
        # by device.
        self._attn_implementation = attn_implementation
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=td, attn_implementation=attn_implementation).to(device)
        self._model.eval()
        self._max_new_tokens = max_new_tokens
        self._verbose = verbose
        self._revision = getattr(self._model.config, "_commit_hash", None) or "unpinned"

    def generate(self, x: str, capture_layers: list[int] | None = None) -> Generation:
        torch = self._torch
        inputs = self._tok.apply_chat_template(
            [{"role": "user", "content": x}], add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(self._device)
        prompt_len = inputs["input_ids"].shape[1]

        t = time.time()
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                return_dict_in_generate=bool(capture_layers),
                output_hidden_states=bool(capture_layers),
            )
        out_ids = generated.sequences if capture_layers else generated
        full = out_ids[0]                                    # prompt + generated answer
        answer = self._tok.decode(full[prompt_len:], skip_special_tokens=True).strip()

        acts = None
        if capture_layers:
            # Each generation step exposes the decision state that produced that step's
            # emitted token. Retaining those states couples the tap view to the answer's
            # actual execution; no replay or second model invocation is permitted.
            steps = generated.hidden_states
            if not steps or len(steps) != full.shape[0] - prompt_len:
                raise ValueError("generation did not expose one tap view per emitted token")
            stacked = torch.stack([
                torch.stack([states[layer][0, -1] for layer in capture_layers], dim=0)
                for states in steps
            ], dim=0)  # [generated token, layer, embedding]
            acts = stacked.float().cpu().numpy()

        if self._verbose:
            shp = None if acts is None else acts.shape
            print(f"    program: {time.time() - t:.1f}s | acts {shp} | answer: {answer[:50]!r}",
                  flush=True)
        return Generation(thought="", answer=answer, activations=acts)

    def digest(self) -> str:
        # Evaluation metadata; certification hashes the actual provisioned model files.
        return (f"gemma2-program-v1:transformers:{self.model_id}@{self._revision}"
                f":attn={self._attn_implementation}")
