"""Shared TreeCut generator: matched answerable / unanswerable math items.

Used by more than one evaluation (confidence's environments, and basic_mocked's
gaming demo + fixture), so it lives here as a shared utility. The purpose-specific
*environments* built on top of it live under `evals/confidence/`.

    SOURCE (imported, not copied): git submodule third_party/treecut-math
      repo:    https://github.com/j-bagel/treecut-math   (Apache-2.0)
      paper:   Ouyang, "TreeCut: A Synthetic Unanswerable Math Word Problem
               Dataset for LLM Hallucination Evaluation", arXiv:2502.13442
               https://arxiv.org/abs/2502.13442
      dataset: https://huggingface.co/datasets/jouyang/treecut-math
    We import upstream's `gen_data` unchanged. It uses flat intra-package imports
    (`from dependency_tree import ...`), so we put its own `treecut/` directory on
    sys.path exactly as upstream's generate_data_file.py does. Run
    `git submodule update --init` to populate it.

Label semantics (application A_app = "the program does not assert answers it
cannot support"): z = 0 answerable; z = 1 unanswerable (TreeCut cut its solution
path, so a confident numeric answer is a hallucination).
"""

from __future__ import annotations

import pathlib
import random
import sys

_TREECUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "third_party" / "treecut-math" / "treecut"
if not (_TREECUT_DIR / "gen_data.py").exists():
    raise ImportError(
        f"TreeCut submodule not found at {_TREECUT_DIR}. "
        "Run: git submodule update --init third_party/treecut-math"
    )
if str(_TREECUT_DIR) not in sys.path:
    sys.path.insert(0, str(_TREECUT_DIR))
import gen_data as _treecut_gen  # noqa: E402  (upstream module, from the submodule)

generate_qa = _treecut_gen.generate_qa


def generate_items(
    n: int,
    seed: int,
    *,
    theme: str = "food",
    num_vars: int = 7,
    ans_depth: int = 3,
    composite_name: bool = False,
    order: str = "random",
) -> list[dict]:
    """A seeded, label-balanced list of TreeCut items: {x, z, answer}.

    Half answerable (z=0), half unanswerable (z=1, one edge cut mid solution path).
    Reproducible from `seed` alone (TreeCut draws from the global `random`).
    """
    if ans_depth < 2:
        raise ValueError("ans_depth must be >= 2 so an unanswerable twin can cut an edge")
    random.seed(seed)
    cut_depth = max(1, ans_depth // 2)  # mid-path cut (TreeCut's max-hallucination position)
    half = n // 2
    items: list[dict] = []
    for k in range(n):
        if k < half:
            qa = generate_qa(theme, composite_name, num_vars, ans_depth, order, False)
            items.append({"x": qa["problem"], "z": 0, "answer": qa["answer"]})
        else:
            qa = generate_qa(theme, composite_name, num_vars, ans_depth, order, True, cut_depth)
            items.append({"x": qa["problem"], "z": 1, "answer": qa["answer"]})  # answer == "unknown"
    random.shuffle(items)
    return items
