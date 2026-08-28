"""The real program (Gemma 4) for the confidence evaluation, thinking ON.

Behind `ct_core.interfaces.ProgramRunner`, so it drops into the certified pipeline
in place of the mock. Model specifics live in `util/model_backend`. NOT exercised on
real weights in CI -- run `python -m evals_runner taps confidence` on a machine with
the model.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ct_core.interfaces import Generation, ProgramRunner

if TYPE_CHECKING:
    from util.model_backend import InferenceBackend


class GemmaProgram(ProgramRunner):
    def __init__(self, backend: "InferenceBackend", max_new_tokens: int = 512,
                 verbose: bool = False):
        self._backend = backend
        self._max_new_tokens = max_new_tokens
        self._verbose = verbose

    def generate(self, x: str, capture_layers: list[int] | None = None) -> Generation:
        if self._verbose:
            print(f"    program: generating (<= {self._max_new_tokens} tok) for: "
                  f"{x.replace(chr(10), ' ')[:70]}...", flush=True)
        t = time.time()
        out = self._backend.generate([{"role": "user", "content": x}], enable_thinking=True,
                                     max_new_tokens=self._max_new_tokens)
        if self._verbose:
            print(f"    program: {time.time() - t:.1f}s | thought {len(out.thought)} chars "
                  f"| answer: {out.answer.replace(chr(10), ' ')[:70]}", flush=True)
        # Same committed program, same enclave-side execution: if a probe tap needs
        # activations, capture them from this backend now (not from a separate model).
        acts = self._backend.residual_activations(x, capture_layers) if capture_layers else None
        return Generation(thought=out.thought, answer=out.answer, activations=acts)

    def digest(self) -> str:
        # backend.digest() carries the checkpoint identity (model id + resolved revision).
        return f"gemma-program-v1:{self._backend.digest()}"
