#!/usr/bin/env python3
"""Make Apollo's Gemma loader work on a CUDA VM. Idempotent; run from the repo root.

RUNBOOK Phase 2 Step A patches this same function for Apple Silicon (pin CPU +
float32). A CUDA host needs a different set of edits, so they live here as code
rather than as prose:

  1. drop `cache_dir=Path("/data/huggingface")` -- Apollo's cluster path, absent here;
  2. drop `local_files_only=True`   -- it blocks the download of the gated checkpoint;
  3. load in bfloat16 rather than float16.

(3) is the one that changes results rather than just letting the load succeed.
Upstream picks `torch.float16 if model_name.size < 9`, so `gemma-2b` would train in
fp16. Gemma-2's activations are large enough to overflow fp16 (the same reason the
RUNBOOK rules float16 out on a Mac), and an overflow here is silent: it lands in the
activations the probe is fitted on, not in an exception. bfloat16 has fp32's exponent
range, is native on the L4 (sm_89), and matches the precision the evaluation then
loads the same weights at -- so the probe is fitted and applied in one dtype.

`device_map="auto"` and `torch_dtype=dtype` are left alone: on a single-GPU host
"auto" places the whole model on the L4, which is what we want.

Every edit is scoped to `get_gemma_model_and_tokenizer`. The mistral and llama
loaders below it carry the same cluster-specific lines and are deliberately left
untouched -- nothing on this path calls them, and narrowing the blast radius keeps
the diff against an unlicensed upstream as small as it can be.
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path("third_party/deception-detection/deception_detection/models.py")
ACTIVATIONS = Path("third_party/deception-detection/deception_detection/activations.py")
FUNCTION = "def get_gemma_model_and_tokenizer("
MARKER = "# Patched for the CUDA port"

EDITS = [
    (
        "drop the cluster cache_dir and local_files_only",
        """        torch_dtype=dtype,
        cache_dir=Path("/data/huggingface"),
        local_files_only=True,
    )""",
        """        torch_dtype=dtype,
    )""",
    ),
    (
        "load bfloat16 instead of float16 below 9B",
        """    # Determine the appropriate dtype based on model size
    dtype = torch.float16 if model_name.size < 9 else torch.bfloat16""",
        """    # Patched for the CUDA port: bfloat16 at every size, not float16 below 9B.
    # Gemma-2's activations overflow float16, and the overflow is silent -- it ends up
    # in the activations the probe is fitted on. bfloat16 keeps fp32's exponent range,
    # is native on the L4, and is the dtype the evaluation loads these weights at.
    dtype = torch.bfloat16""",
    ),
]


def gemma_function(source: str) -> tuple[int, int]:
    """Byte range of `get_gemma_model_and_tokenizer` alone, so sibling loaders are safe."""
    start = source.index(FUNCTION)
    following = source.find("\ndef ", start + len(FUNCTION))
    return start, len(source) if following == -1 else following


def patch_batch_size() -> bool:
    """Make the activation batch size settable, for cards smaller than Apollo's.

    `PairedActivations.from_dataset` hardcodes batch_size=12. That is fine for a 2B
    model on a 24 GB card, but 9B in bfloat16 is 18.5 GB of weights before any
    activations exist, and a 12-dialogue batch with output_hidden_states across 42
    layers does not fit in what is left. Reading the default from the environment
    keeps the upstream behaviour when the variable is unset.
    """
    if not ACTIVATIONS.exists():
        print(f"  missing {ACTIVATIONS}", file=sys.stderr)
        return False
    source = ACTIVATIONS.read_text()
    if "CT_ACT_BATCH_SIZE" in source:
        print("  already applied: configurable activation batch size")
        return True
    old = """        model: PreTrainedModel | None,
        batch_size: int = 12,"""
    new = """        model: PreTrainedModel | None,
        batch_size: int = int(os.environ.get("CT_ACT_BATCH_SIZE", "12")),"""
    if old not in source:
        print("  FAILED to locate the from_dataset batch_size default", file=sys.stderr)
        return False
    source = source.replace(old, new, 1)
    if "\nimport os" not in source:
        source = source.replace("\nimport torch", "\nimport os\nimport torch", 1)
    ACTIVATIONS.write_text(source)
    print("  applied: activation batch size reads CT_ACT_BATCH_SIZE (default 12)")
    return True


def main() -> int:
    if not TARGET.exists():
        print(f"missing {TARGET} -- check out the submodule first", file=sys.stderr)
        return 1
    source = TARGET.read_text()
    try:
        start, end = gemma_function(source)
    except ValueError:
        print(f"{FUNCTION} not found in {TARGET}", file=sys.stderr)
        return 1
    body = source[start:end]
    already = MARKER in body

    for label, old, new in EDITS:
        if old in body:
            body = body.replace(old, new, 1)
            print(f"  applied: {label}")
        elif already:
            print(f"  already applied: {label}")
        else:
            print(f"  FAILED to locate: {label}", file=sys.stderr)
            print("  upstream models.py has moved; re-derive the patch", file=sys.stderr)
            return 1

    patched = source[:start] + body + source[end:]
    if patched != source:
        TARGET.write_text(patched)
    # The gemma loader must end up with neither cluster line and with bf16.
    start, end = gemma_function(TARGET.read_text())
    body = TARGET.read_text()[start:end]
    for forbidden in ('cache_dir=Path("/data/huggingface")', "local_files_only=True",
                      "torch.float16"):
        if forbidden in body:
            print(f"  post-check failed: {forbidden} still in the gemma loader",
                  file=sys.stderr)
            return 1
    print("  gemma loader verified: no cluster paths, bfloat16 only")
    if not patch_batch_size():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
