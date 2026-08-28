#!/usr/bin/env python3
"""Regenerate the committed basic-mocked TreeCut fixture (deterministic).

    python -m evals.basic_mocked.make_fixture
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evals.basic_mocked import FIXTURE     # noqa: E402
from util.treecut import generate_items    # noqa: E402

N = 16
SEED = 7


def main() -> None:
    items = generate_items(N, seed=SEED)
    FIXTURE.parent.mkdir(exist_ok=True)
    with FIXTURE.open("w") as f:
        for it in items:
            f.write(json.dumps(it, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"wrote {len(items)} TreeCut items (seed={SEED}) -> {FIXTURE}")


if __name__ == "__main__":
    main()
