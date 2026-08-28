"""The program under test for the conflict evaluation, thinking ON.

Mirrors `evals/confidence/program.py`; each eval owns its own program digest, which
is part of h_c. Model specifics live in `util/model_backend`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ct_core.interfaces import Generation, ProgramRunner

if TYPE_CHECKING:
    from util.model_backend import InferenceBackend


class BookingProgram(ProgramRunner):
    def __init__(self, backend: "InferenceBackend", max_new_tokens: int = 512,
                 verbose: bool = False):
        self._backend = backend
        self._max_new_tokens = max_new_tokens
        self._verbose = verbose

    def generate(self, x: str, capture_layers: list[int] | None = None) -> Generation:
        t = time.time()
        out = self._backend.generate([{"role": "user", "content": x}],
                                     enable_thinking=True,
                                     max_new_tokens=self._max_new_tokens)
        if self._verbose:
            print(f"    program: {time.time() - t:.1f}s | thought {len(out.thought)} chars "
                  f"| answer: {out.answer.replace(chr(10), ' ')[:70]}", flush=True)
        return Generation(thought=out.thought, answer=out.answer)

    def digest(self) -> str:
        return f"conflict-program-v1:{self._backend.digest()}"
